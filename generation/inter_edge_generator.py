import argparse
import logging
import math

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import _xavier_init_param, _xavier_linear_init, NodeEncoder

logger = logging.getLogger(__name__)


class ClusterConditioner(nn.Module):
    def __init__(self, K: int, pool_dim: int, emb_dim: int):
        super().__init__()

        self.to_gamma = nn.Sequential(
            nn.Linear(K + pool_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        )

        self.to_beta = nn.Sequential(
            nn.Linear(K + pool_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        )

    def forward(self, z, S, x_pool_c):
        ctx = x_pool_c.unsqueeze(0).expand(z.shape[0], -1)

        inp = torch.cat([S, ctx], dim=1)

        gamma = torch.tanh(self.to_gamma(inp))
        beta  = self.to_beta(inp)

        return z * (1 + gamma) + beta


class ExternalNodeEncoder(nn.Module):
    """
    Encode external nodes. Because we dont have access to their features.
    """
    def __init__(self, K: int, pool_dim: int, emb_dim: int, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(K + pool_dim, emb_dim),
            nn.LayerNorm(emb_dim),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(emb_dim, emb_dim),
            nn.LayerNorm(emb_dim),
            nn.GELU(),
        )

    def forward(self, S_rows: torch.Tensor, x_pool_c: torch.Tensor) -> torch.Tensor:
        """
        S_rows : [N_ext, K]
        x_pool_c : [pool_dim] or [N_ext, pool_dim]
        """
        if x_pool_c.dim() == 1:
            x_pool_c = x_pool_c.unsqueeze(0).expand(S_rows.shape[0], -1)
        return self.proj(torch.cat([S_rows, x_pool_c], dim=1))


def masked_bce_loss(logits, labels, mask):
    loss = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
    return (loss * mask.float()).sum() / mask.float().sum().clamp(min=1)


def masked_kl_bernoulli(logits_p, logits_q, mask, eps: float = 1e-6):
    """KL( p || q ) between Bernoulli(sigmoid(logits_p)) and Bernoulli(sigmoid(logits_q)) averaged over mask positions. """
    p = torch.sigmoid(logits_p).clamp(eps, 1 - eps)
    q = torch.sigmoid(logits_q).clamp(eps, 1 - eps)
    kl = p * (p.log() - q.log()) + (1 - p) * ((1 - p).log() - (1 - q).log())
    return (kl * mask.float()).sum() / mask.float().sum().clamp(min=1)


def soft_entropy(S: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Entropy of each node's soft assignment"""
    p = S.clamp(min=eps)
    p = p / p.sum(dim=1, keepdim=True).clamp(min=eps)
    return -(p * p.log()).sum(dim=1)


def topk_bridge_candidates(S: torch.Tensor, c: int, s_threshold: float, topk: int, H: Optional[torch.Tensor] = None, exclude_cluster: Optional[int] = None) -> torch.Tensor:
    """
    Candidates for cross-community edges: among nodes with membership in cluster 'c' above 's_threshold', the top-k by assignment entropy.

    Entropy is simply converting the soft assignments into probabilities, so converting is the way we suggest in the paer.

    H: precomputed soft entropy of S, if available. But we never precomputed.
    """
    keep = S[:, c] > s_threshold
    if exclude_cluster is not None:
        # Drop non bridge nodes.
        keep = keep & (S.argmax(dim=1) != exclude_cluster)
    above = keep.nonzero(as_tuple=True)[0]
    if above.numel() == 0:
        return above
    if H is None:
        H = soft_entropy(S)
    vals = H[above]
    if vals.numel() > topk:
        _, li = vals.topk(topk)
        return above[li]
    return above


def soft_cluster_affinity(s_src_c1, S_dst_all, a_row, eps=1e-8):
    a = a_row.clamp(min=0)
    a = a / a.sum().clamp(min=eps) # c2 distribution
    dst_term = (S_dst_all * a.unsqueeze(0)).sum(-1) # [N2] in [0,1]
    return s_src_c1.unsqueeze(1) * dst_term.unsqueeze(0) # [N1, N2] in [0,1]


def _log_prior(prior: torch.Tensor, eps: float = 1e-8, min_log: float = -10.0) -> torch.Tensor:
    """Numerically stable log of a [0, 1]-ish prior score."""
    return (prior.clamp(0, 1) + eps).log().clamp(min=min_log)


def budget_select(scores: torch.Tensor, logit_offset: float = 0.0, cap: Optional[float] = None):
    """
    Pick edges from a [N1, N2] score matrix using the model's own expected edge count as the budget.
    
    Returns (row_idx, col_idx, probs_of_selected).
    """
    # Training subsamples negatives at rate s, so model odds overstate the true odds by 1/s; logit_offset = log(s) corrects for this.
    probs = torch.sigmoid(scores + logit_offset)
    flat = probs.reshape(-1)
    budget = int(flat.sum().round().clamp(min=0, max=flat.numel()).item())
    if cap is not None:
        budget = min(budget, int(max(cap, 0)))
    if budget == 0:
        empty = torch.zeros(0, dtype=torch.long, device=scores.device)
        return empty, empty, torch.zeros(0, device=scores.device)
    top = flat.topk(budget)
    return top.indices // probs.shape[1], top.indices % probs.shape[1], top.values


class BaseInterEdgeGenerator(nn.Module):
    def __init__(self, args: argparse.Namespace, feat_dim: int, pool_dim: int, K: int):
        super().__init__()
        self.args = args
        self.feat_dim = feat_dim
        self.pool_dim = pool_dim
        self.K = K
        self.weighted = args.weighted
        self.emb_dim = args.edge_emb_dim
        self.dropout = args.dropout
        self.prior_weight = args.prior_weight
        self.kl_coef = args.prior_kl_coef
        self.s_threshold = args.s_threshold
        self.topk_nodes = args.topk_nodes

        frac = args.inter_neg_pool_frac
        self.logit_offset = math.log(frac) if frac > 0 else 0.0

        self.encoder = NodeEncoder(feat_dim, self.emb_dim, self.dropout)

        self.cluster_cond = ClusterConditioner(K, pool_dim, self.emb_dim)
        self.ext_encoder = ExternalNodeEncoder(K, pool_dim, self.emb_dim, self.dropout)

        pw = min(max(self.prior_weight, 1e-4), 1.0 - 1e-4) 
        self.prior_alpha = nn.Parameter(torch.logit(torch.tensor(pw))) 
 

    def encode_nodes(self, x, S, x_pool_c):
        """Local nodes: x: [N, feat_dim], S: [N, K], x_pool_c: [pool_dim]"""
        z = self.encoder(x)
        z = self.cluster_cond(z, S, x_pool_c)
        return z


    def encode_external(self, S_rows, x_pool_c):
        """External nodes without raw features."""
        return self.ext_encoder(S_rows, x_pool_c)


    def _build_per_pair_prior(self,
            pairs: torch.Tensor, # [E, 2] (src_local, dst_parent_local)
            S_local: torch.Tensor, # [N, K]
            S_pool_parent: torch.Tensor, # [N_parent, K]
            A_pool: torch.Tensor, # [K, K]
            c1: int,
            mask: torch.Tensor, # [E] bool
        ) -> torch.Tensor:
        """
        Build a soft cluster-affinity prior for each pair. Aka, information coming from the hierarchy

        prior = S_local[src, c1] * sum_k(A_pool[c1, k] * S_parent[dst, k])

        Returns Soft cluster-affinity prior score.
        """
        E = pairs.shape[0]
        device = pairs.device
        N = S_local.shape[0]
        N_par = S_pool_parent.shape[0]

        src = pairs[:, 0].clamp(0, N - 1)
        dst = pairs[:, 1].clamp(0, N_par - 1)

        if E == 0:
            return torch.zeros(0, device=device)

        s_src = S_local[src, c1] # [E]
        s_dst_all = S_pool_parent[dst] # [E, K]

        a = A_pool[c1].clamp(min=0)
        a = a / a.sum().clamp(min=1e-8) # [K]
        dst_term = (s_dst_all * a.unsqueeze(0)).sum(-1) # [E]
        prior = (s_src * dst_term) * mask.float() # [E] in [0,1]
        return prior


    def _loss_from_pairs(self,
            pos_logits: torch.Tensor, neg_logits: torch.Tensor,
            pos_mask: torch.Tensor, neg_mask: torch.Tensor,
            prior_pos: Optional[torch.Tensor] = None,
            prior_neg: Optional[torch.Tensor] = None,
        ):
        alpha = torch.sigmoid(self.prior_alpha)
        pos_logits = pos_logits + alpha * _log_prior(prior_pos)
        neg_logits = neg_logits + alpha * _log_prior(prior_neg)

        scores = torch.cat([pos_logits, neg_logits], dim=1)
        labels = torch.cat([torch.ones_like(pos_logits), torch.zeros_like(neg_logits)], dim=1)
        mask = torch.cat([pos_mask, neg_mask], dim=1)
        bce = masked_bce_loss(scores, labels, mask)

        return bce


    def loss(self, batch):
        
        raise NotImplementedError


    def generate(self, x, S, x_pool, A_pool, S_pool_parent, x_pool_parent, cluster_id, noise: float = 0.0, return_z: bool = False):
        raise NotImplementedError


    def init_weights(self):
        _xavier_linear_init(self)


class SBilinearInterEdgeScorer(BaseInterEdgeGenerator):
    """
    Bilinear scorer: score_ij = z_i^T W z_j, z_i a local node, z_j an external (parent-level) node. 
    
    alpha * log(prior_ij) is added to the score.
    """

    def __init__(self, args, feat_dim, pool_dim, K):
        super().__init__(args, feat_dim, pool_dim, K)

        self.W = nn.Parameter(torch.empty(self.emb_dim, self.emb_dim))
        _xavier_init_param(self.W)

        if self.weighted:
            self.weight_head = nn.Sequential(
                nn.Linear(self.emb_dim * 2, self.emb_dim),
                nn.GELU(),
                nn.Linear(self.emb_dim, 1),
                nn.Softplus(),
            )

    def _score_pair_indices(self,
            z_local: torch.Tensor, # [N, D]
            z_parent: torch.Tensor, # [N_parent, D]
            pairs: torch.Tensor, # [E, 2]
            mask: torch.Tensor, # [E]
        ) -> torch.Tensor:
        """Logits for pairs (row of z_local, row of z_parent)."""

        if pairs.shape[0] == 0:
            return torch.zeros(0, device=pairs.device)

        N = z_local.shape[0]
        N_par = z_parent.shape[0]
        src = pairs[:, 0].clamp(0, max(N - 1, 0))
        dst = pairs[:, 1].clamp(0, max(N_par - 1, 0))

        z_s = z_local[src]
        z_d = z_parent[dst]
        logits = (z_s @ self.W * z_d).sum(dim=-1) # [E]

        return logits 

    def _score_matrix(self, z_local_sub: torch.Tensor, z_ext: torch.Tensor) -> torch.Tensor:
        """[N1, N2] full bipartite score matrix."""
        return (z_local_sub @ self.W) @ z_ext.T
        

    def loss(self, batch):
        x = batch['x']
        S = batch['S']
        x_pool = batch['x_pool']
        pos_pairs = batch['pos_pairs']
        neg_pairs = batch['neg_pairs']
        pos_mask = batch['pos_mask']
        neg_mask = batch['neg_mask']
        node_mask = batch['node_mask']
        A_pool = batch['A_pool']
        S_pool_parent = batch['S_pool_parent']
        x_pool_parent = batch['x_pool_parent']
        cluster_ids = batch['cluster_id']

        B = x.shape[0]
        pos_logits = torch.zeros_like(pos_mask, dtype=torch.float)
        neg_logits = torch.zeros_like(neg_mask, dtype=torch.float)
        prior_pos = torch.zeros_like(pos_mask, dtype=torch.float)
        prior_neg = torch.zeros_like(neg_mask, dtype=torch.float)

        for b in range(B):
            N = int(node_mask[b].sum().item())
            c1 = int(cluster_ids[b].item())
            x_b = x[b, :N]
            S_b = S[b, :N]
            xp = x_pool[b]

            Spar = S_pool_parent[b]
            xppar = x_pool_parent[b]
            N_par = Spar.shape[0]

            # local side: encode only referenced src rows. z_local is used only via z_local[src], 
            # so this leaves the loss unchanged and avoids O(N) activations on large internal subgraphs
            src_all = torch.cat([pos_pairs[b][pos_mask[b], 0], neg_pairs[b][neg_mask[b], 0], ]).clamp(0, max(N - 1, 0))
            needed_l = torch.unique(src_all)
            if needed_l.numel() == 0:
                needed_l = torch.zeros(1, dtype=torch.long, device=x_b.device)
            remap_l = torch.zeros(N, dtype=torch.long, device=x_b.device)
            remap_l[needed_l] = torch.arange(needed_l.numel(), device=x_b.device)

            z_local = self.encode_nodes(x_b[needed_l], S_b[needed_l], xp)

            # parent side: encode only referenced dst rows 
            dst_all = torch.cat([pos_pairs[b][pos_mask[b], 1], neg_pairs[b][neg_mask[b], 1], ]).clamp(0, max(N_par - 1, 0))
            needed_p = torch.unique(dst_all)
            if needed_p.numel() == 0:
                needed_p = torch.zeros(1, dtype=torch.long, device=Spar.device)
            remap_p = torch.zeros(N_par, dtype=torch.long, device=Spar.device)
            remap_p[needed_p] = torch.arange(needed_p.numel(), device=Spar.device)

            Spar_sub = Spar[needed_p]
            c2_sub = Spar_sub.argmax(dim=-1).clamp(0, xppar.shape[0] - 1)
            z_parent = self.encode_external(Spar_sub, xppar[c2_sub])

            # remapped pair copies for scoring, leave original pairs unchanged for prior computation
            pos_b = pos_pairs[b].clone()
            neg_b = neg_pairs[b].clone()
            pos_b[:, 0] = remap_l[pos_b[:, 0].clamp(0, max(N - 1, 0))]
            neg_b[:, 0] = remap_l[neg_b[:, 0].clamp(0, max(N - 1, 0))]
            pos_b[:, 1] = remap_p[pos_b[:, 1].clamp(0, max(N_par - 1, 0))]
            neg_b[:, 1] = remap_p[neg_b[:, 1].clamp(0, max(N_par - 1, 0))]

            pos_logits[b] = self._score_pair_indices(z_local, z_parent, pos_b, pos_mask[b])
            neg_logits[b] = self._score_pair_indices(z_local, z_parent, neg_b, neg_mask[b])

            pp = self._build_per_pair_prior(pos_pairs[b], S_b, Spar, A_pool[b], c1, pos_mask[b])
            pn = self._build_per_pair_prior(neg_pairs[b], S_b, Spar, A_pool[b], c1, neg_mask[b])
            prior_pos[b] = pp
            prior_neg[b] = pn

        return self._loss_from_pairs(pos_logits, neg_logits, pos_mask, neg_mask, prior_pos, prior_neg)


    @torch.no_grad()
    def generate(self, x, S, x_pool, A_pool, S_pool_parent, x_pool_parent, cluster_id, noise: float = 0.0, return_z: bool = False):
        """
        x : [N, feat_dim]
        S : [N, K]
        x_pool : [pool_dim]
        A_pool : [K, K]
        S_pool_parent : [N_parent, K]
        x_pool_parent : [K, pool_dim]
        return_z : if True, also return z_local [N, emb_dim] as third element of the tuple. No longer using this, it was for another architectures that we dropped.

        Returns (edge_index, edge_weight) or (edge_index, edge_weight, z_local).
        """
        self.eval()
        device = S.device

        all_src, all_dst, all_w = [], [], []

        connected = (A_pool[cluster_id] > self.args.inter_gen_cluster_threshold).nonzero(as_tuple=True)[0].tolist() 
        H_loc = soft_entropy(S)
        H_par = soft_entropy(S_pool_parent)
        local_cands = topk_bridge_candidates(S, cluster_id, self.s_threshold, self.topk_nodes, H=H_loc)

        if len(local_cands) == 0: 
            ei_out = torch.zeros(2, 0, dtype=torch.long, device=device)
            z_empty = torch.zeros(0, self.emb_dim, device=device)
            return (ei_out, None, z_empty) if return_z else (ei_out, None)

        # Encode only the candidate rows. 'x' is on CPU for large graphs, so we index first then move to GPU.
        cands_cpu = local_cands.cpu() if local_cands.is_cuda else local_cands
        x_cand = x[cands_cpu].to(device) if x.device != device else x[local_cands]
        z_loc_all = self.encode_nodes(x_cand, S[local_cands], x_pool)

        for c2 in connected:
            if c2 == cluster_id or c2 >= S_pool_parent.shape[1]:
                continue

            ext_cands = topk_bridge_candidates(S_pool_parent, c2, self.s_threshold, self.topk_nodes, H=H_par, exclude_cluster=cluster_id)
            if len(ext_cands) == 0:
                continue

            S_ext = S_pool_parent[ext_cands] # [N_ext, K]
            xp_c2 = x_pool_parent[c2] if c2 < x_pool_parent.shape[0] else x_pool
            z_ext = self.encode_external(S_ext, xp_c2) # [N_ext, D]
            z_loc = z_loc_all # [N_loc, D], rows aligned with local_cands

            scores = self._score_matrix(z_loc, z_ext)

            logger.info(f"[cal] {tuple(scores.shape)} p_mean={torch.sigmoid(scores).mean():.4f} p>0.5={float((torch.sigmoid(scores)>0.5).float().mean()):.4f}")

            prior = soft_cluster_affinity(S[local_cands, cluster_id], S_pool_parent[ext_cands], A_pool[cluster_id])
            alpha = torch.sigmoid(self.prior_alpha)
            scores = scores + alpha * _log_prior(prior)

            if noise > 0:
                scores = scores + noise * torch.randn_like(scores)

            # inter_budget_scale = 0 disables the ceiling.
            scale = self.args.inter_budget_scale
            cap = float(A_pool[cluster_id, c2]) * scale if scale > 0 else None
            li, ei, _ = budget_select(scores, self.logit_offset, cap=cap)
            if li.numel() == 0:
                continue

            all_src.append(local_cands[li])
            all_dst.append(ext_cands[ei])

            if self.weighted:
                z_s = z_loc[li]
                z_d = z_ext[ei]
                w = self.weight_head(torch.cat([z_s, z_d], dim=-1)).squeeze(-1)
                all_w.append(w)

        if not all_src:
            ei_out = torch.zeros(2, 0, dtype=torch.long, device=device)
            ew_out = None
            return (ei_out, ew_out, z_loc_all) if return_z else (ei_out, ew_out)

        edge_index = torch.stack([torch.cat(all_src), torch.cat(all_dst)], dim=0)
        edge_weight = torch.cat(all_w) if (self.weighted and all_w) else None
        if return_z:
            return edge_index, edge_weight, z_loc_all
        return edge_index, edge_weight

    def init_weights(self):
        _xavier_linear_init(self)
        _xavier_init_param(self.W)
