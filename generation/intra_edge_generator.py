import argparse
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, Batch

from utils import _xavier_init_param, _xavier_linear_init, NodeEncoder



class DegreeConditioner(nn.Module):
    def __init__(self, emb_dim: int):
        super().__init__()
        self.to_gamma = nn.Linear(2, emb_dim)
        self.to_beta = nn.Linear(2, emb_dim)

    def forward(self, z, degree):
        deg = torch.log1p(degree) 

        gamma = torch.tanh(self.to_gamma(deg))
        beta = self.to_beta(deg) 

        return z * (1 + gamma) + beta


def masked_bce_loss(scores, labels, mask):
    mask = mask.float()
    loss = F.binary_cross_entropy_with_logits(scores, labels, reduction='none')
    return (loss * mask).sum() / mask.sum().clamp(min=1)


def masked_weight_loss(pred_w, true_w, mask, delta: float = 1.0):
    mask = mask.float()
    loss = F.huber_loss(pred_w, true_w, reduction='none', delta=delta) # might check delta value later
    return (loss * mask).sum() / mask.sum().clamp(min=1)


def _finalize_undirected(src: torch.Tensor, dst: torch.Tensor, ew: Optional[torch.Tensor], N: int,) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Union (i,j) and (j,i), keeping the first occurrence of each undirected edge. """

    src_u = torch.cat([src, dst])
    dst_u = torch.cat([dst, src])
    ew_u = torch.cat([ew, ew]) if ew is not None else None

    lo = torch.minimum(src_u, dst_u)
    hi = torch.maximum(src_u, dst_u)
    key = lo * N + hi

    order = torch.argsort(key, stable=True)
    key_sorted = key[order]
    first = torch.ones_like(key_sorted, dtype=torch.bool)
    first[1:] = key_sorted[1:] != key_sorted[:-1]
    keep = order[first]

    s1 = src_u[keep]
    d1 = dst_u[keep]

    edge_index = torch.stack([torch.cat([s1, d1]), torch.cat([d1, s1])], dim=0)
    if ew_u is not None:
        e1 = ew_u[keep]
        return edge_index, torch.cat([e1, e1])
    
    return edge_index, None


@torch.no_grad()
def streaming_degree_topk(N: int, degree: torch.Tensor, # [N, 2]
        directed: bool, weighted: bool, degree_mode: str,
        noise: float, chunk_size: int, device: torch.device,
        logits_fn: Callable[[int, int], torch.Tensor], # (start, end) -> [C, N] logits for pairs (rows, all)
        logits_rev_fn: Optional[Callable[[int, int], torch.Tensor]] = None,  # (start, end) -> [C, N], entry [c, j] = logit(j -> start+c)
        weight_fn: Optional[Callable[[int, int], torch.Tensor]] = None, # (start, end) -> [C, N]
        weight_rev_fn: Optional[Callable[[int, int], torch.Tensor]] = None,  # (start, end) -> [C, N], entry [c, j] = weight(j -> start+c)
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Degree-constrained edge selection, streamed over chunks of source rows:
    compute logits, add noise noise, sigmoid, mask self-loops, optionally max-symmetrize against the reverse direction (undirected),
    and keep each row's top-k. Selection is row-local, so chunking does not change the result.

    Per-row budget k:
      directed : degree[:, 1] -> we pretty much always taking out degree through code.
      undirected, degree_mode='undirected' : ceil(degree[:, 0] / 2) -> so that the total number of edges is at most degree[:, 0] for undirected graphs. 

    """
    assert degree_mode in ('directed', 'undirected'), f"degree_mode must be 'directed' or 'undirected', got {degree_mode}"

    # per-row budgets
    if directed:
        k_per_row = degree[:, 1].long()
        symmetrize_out = False
    else:
        assert logits_rev_fn is not None, "undirected streaming top-k needs logits_rev_fn"
        if degree_mode == 'undirected':
            k_per_row = (degree[:, 0].float() / 2.0).ceil().long()
        else:
            k_per_row = degree[:, 1].long()
        symmetrize_out = True
    k_per_row = k_per_row.clamp(min=0, max=max(N - 1, 0))

    k_max = int(k_per_row.max().item()) if N > 0 else 0
    k_max = min(k_max, N - 1)

    if k_max <= 0 or N == 0:
        ei = torch.zeros(2, 0, dtype=torch.long, device=device)
        ew = torch.zeros(0, device=device) if weighted else None
        return ei, ew

    use_weights = weighted and weight_fn is not None
    
    pair_budget = 1_000_000
    chunk_size = max(1, min(chunk_size, pair_budget // max(N, 1)))

    src_parts, dst_parts, ew_parts = [], [], []

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        C = end - start

        logits = logits_fn(start, end) # [C, N]
        if noise > 0:
            logits = logits + torch.randn_like(logits) * noise
        probs = torch.sigmoid(logits)
        del logits

        w_blk = weight_fn(start, end) if use_weights else None # [C, N]

        if symmetrize_out:
            logits_r = logits_rev_fn(start, end) # [C, N]
            if noise > 0:
                logits_r = logits_r + torch.randn_like(logits_r) * noise
            probs = torch.maximum(probs, torch.sigmoid(logits_r))
            del logits_r
            if w_blk is not None and weight_rev_fn is not None:
                w_blk = 0.5 * (w_blk + weight_rev_fn(start, end))

        # self-loops
        row_ids = torch.arange(start, end, device=device)
        probs[torch.arange(C, device=device), row_ids] = float('-inf')

        # per-row top-k within the chunk (row-local, hence exact)
        _, topk_idx = probs.topk(k_max, dim=1)                        # [C, k_max]
        del probs
        arange = torch.arange(k_max, device=device).unsqueeze(0)
        valid = arange < k_per_row[start:end].unsqueeze(1)            # [C, k_max]

        rows = row_ids.unsqueeze(1).expand_as(topk_idx)
        s = rows[valid]
        d = topk_idx[valid]
        e = w_blk.gather(1, topk_idx)[valid] if w_blk is not None else None
        if w_blk is not None:
            del w_blk

        non_self = s != d
        src_parts.append(s[non_self])
        dst_parts.append(d[non_self])
        if e is not None:
            ew_parts.append(e[non_self])

    src = torch.cat(src_parts)
    dst = torch.cat(dst_parts)
    ew = torch.cat(ew_parts) if ew_parts else None

    if symmetrize_out:
        return _finalize_undirected(src, dst, ew, N)

    return torch.stack([src, dst], dim=0), (ew if weighted else None)


def _mlp_pair_block(head: nn.Module, z_rows: torch.Tensor, z_all: torch.Tensor) -> torch.Tensor:
    """Apply a pair MLP to all ordered pairs (rows, all): [C, N]."""
    C, D = z_rows.shape
    N = z_all.shape[0]
    pair = torch.cat([
        z_rows.unsqueeze(1).expand(C, N, D),
        z_all.unsqueeze(0).expand(C, N, D),
    ], dim=-1)
    return head(pair).squeeze(-1)


def _mlp_pair_block_rev(head: nn.Module, z_rows: torch.Tensor, z_all: torch.Tensor) -> torch.Tensor:
    """
    Apply a pair MLP to all ordered pairs (all, rows), 
    transposed to [C, N]: entry [c, j] = head(j -> row c)."""
    C, D = z_rows.shape
    N = z_all.shape[0]
    pair = torch.cat([z_all.unsqueeze(1).expand(N, C, D), z_rows.unsqueeze(0).expand(N, C, D), ], dim=-1)
    return head(pair).squeeze(-1).T.contiguous()


class BaseEdgeGenerator(nn.Module):
    # Similar to BaseNodeGenerator, but for intra edges.
    # If you implement a new edge generator, subclass this and implement forward, loss, and generate.
    def __init__(self, args: argparse.Namespace, feat_dim: int):
        super().__init__()
        self.args = args
        self.feat_dim = feat_dim
        self.directed = args.directed
        self.weighted = args.weighted
        self.weight_coef = args.weight_coef
        self.emb_dim = args.edge_emb_dim
        self.dropout = args.dropout
        self.encoder = NodeEncoder(feat_dim, self.emb_dim, self.dropout)
        self.degree_cond = DegreeConditioner(self.emb_dim)


    def compute_loss(self, pos_scores, neg_scores, edge_mask, neg_mask, pos_w_pred=None, edge_weight=None):
        """
        BCE over positives (masked by edge_mask) + negatives (masked by neg_mask).
        Optional Huber loss on positive edge weights.
        """
        scores = torch.cat([pos_scores, neg_scores], dim=1) # [B, max_E + max_neg]
        labels = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores) ], dim=1)
        mask = torch.cat([edge_mask, neg_mask], dim=1) # BUGFIX
        
        bce = masked_bce_loss(scores, labels, mask)

        if self.weighted and pos_w_pred is not None and edge_weight is not None:
            w_loss = masked_weight_loss(pos_w_pred, edge_weight, edge_mask)
            return bce + self.weight_coef * w_loss
        
        return bce

    def generate(self, *args, **kwargs):
        raise NotImplementedError

    def loss(self, *args, **kwargs):
        raise NotImplementedError

    def init_weights(self):
        _xavier_linear_init(self)



class VGAEEdgeGenerator(BaseEdgeGenerator):
    """
    Encoder : project x + degree -> GCN -> (mu, logvar)
    Decoder : bilinear score z_i^T W z_j on sampled z -> edge logits

    Training : PyG Batch (disjoint-union) for parallel per-graph message passing.
    Generation: per-graph, samples z ~ N(0,I) if edge_index is None, else uses posterior from passed edge_index.
    """
    def __init__(self, args: argparse.Namespace, feat_dim: int):
        super().__init__(args, feat_dim)

        self.kl_weight = args.kl_weight
        z_dim = self.emb_dim

        self.gcn1 = GCNConv(self.emb_dim, self.emb_dim * 2)
        self.gcn_mu = GCNConv(self.emb_dim * 2, z_dim)
        self.gcn_var = GCNConv(self.emb_dim * 2, z_dim)

        self.W = nn.Parameter(torch.empty(z_dim, z_dim))
        _xavier_init_param(self.W) # BUGFIX

        if self.weighted:
            self.weight_mlp = nn.Sequential(
                nn.Linear(z_dim * 2, 64),
                nn.GELU(),
                nn.Linear(64, 1),
                nn.Softplus(),
            )
        else:
            self.weight_mlp = None


    def _project_features(self, x, degree):
        """x,degree: [N, *] (no batch dim) -> z: [N, emb_dim]"""
        z = self.encoder(x.unsqueeze(0)).squeeze(0)
        z = self.degree_cond(z.unsqueeze(0), degree.unsqueeze(0)).squeeze(0)
        return z


    def _encode_single(self, x, degree, edge_index):
        """Single-graph encoder. Used by generate()."""
        z = self._project_features(x, degree)
        h = F.gelu(self.gcn1(z, edge_index))
        mu = self.gcn_mu(h, edge_index)
        logvar = self.gcn_var(h, edge_index)
        logvar = torch.clamp(logvar, -10, 10) # numerical stability
        return mu, logvar

    def _encode_batch(self,
            x: torch.Tensor, # [B, max_N, feat_dim]
            degree: torch.Tensor, # [B, max_N, 2]
            edge_index_list: list, # list[B] of [2, E_b]
            node_mask: torch.Tensor, # [B, max_N] -< True = real
        ):
        """
        Batched encoder via PyG disjoint-union.
        Returns mu_per_graph, logvar_per_graph : list[B] of [N_b, z_dim]
        """
        B = x.shape[0]
        device = x.device

        # build a PyG Batch from real nodes only
        data_list = []
        for b in range(B):
            N_b = int(node_mask[b].sum().item())
            if N_b == 0:
                data_list.append(Data(x = torch.zeros(0, self.emb_dim, device=device), edge_index = torch.zeros(2, 0, dtype=torch.long, device=device),))
                continue
            z_b = self._project_features(x[b, :N_b], degree[b, :N_b]) # [N_b, emb_dim]
            assert (edge_index_list[b].numel() == 0) or (edge_index_list[b].max() < N_b), f"edge_index contains node indices >= number of real nodes: max index {edge_index_list[b].max().item()}, real nodes {N_b}"
            data_list.append(Data(x=z_b, edge_index=edge_index_list[b]))

        big = Batch.from_data_list(data_list)

        # run GCN once on the disjoint union
        h = F.gelu(self.gcn1(big.x, big.edge_index))
        mu = self.gcn_mu(h, big.edge_index)
        logvar = self.gcn_var(h, big.edge_index)
        logvar = torch.clamp(logvar, -10, 10) # numerical stability

        # split back per-graph
        ptr = big.ptr # [B+1]
        mu_list, logvar_list = [], []
        for b in range(B):
            mu_list.append(mu[ptr[b]:ptr[b + 1]])
            logvar_list.append(logvar[ptr[b]:ptr[b + 1]])

        return mu_list, logvar_list


    def _reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std


    def _decode_pairs(self, z, pairs):
        """z: [N, D], pairs: [E, 2] -> scores [E], w_pred [E] or None"""
        z_src = z[pairs[:, 0]]
        z_dst = z[pairs[:, 1]]
        scores = (z_src @ self.W * z_dst).sum(dim=-1)

        w_pred = None
        if self.weight_mlp is not None:
            w_pred = self.weight_mlp(torch.cat([z_src, z_dst], dim=-1)).squeeze(-1)
        return scores, w_pred


    def forward(self, x, degree, pos_pairs, neg_pairs, edge_index_list, node_mask):
        """
        Batched forward via PyG Batch. Pads pos/neg scores back to [B, max_E].
        """
        B = x.shape[0]
        device = x.device

        mu_list, logvar_list = self._encode_batch(x, degree, edge_index_list, node_mask)

        all_pos_s, all_neg_s, all_pos_w = [], [], []
        kl_total = torch.tensor(0.0, device=device)

        for b in range(B):
            mu, logvar = mu_list[b], logvar_list[b]
            if mu.shape[0] == 0:
                # empty graph -> emit zero scores matching the padded shape
                all_pos_s.append(torch.zeros(pos_pairs.shape[1], device=device))
                all_neg_s.append(torch.zeros(neg_pairs.shape[1], device=device))
                if self.weighted:
                    all_pos_w.append(torch.zeros(pos_pairs.shape[1], device=device))
                continue

            z = self._reparameterize(mu, logvar)
            kl_elem = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
            kl_per_node = kl_elem.sum(dim=-1)
            kl = kl_per_node.mean()

            kl_total = kl_total + kl

            assert pos_pairs[b].max() < z.shape[0], f"pos_pairs contains node indices >= number of real nodes: max index {pos_pairs[b].max().item()}, real nodes {z.shape[0]}"
            assert neg_pairs[b].max() < z.shape[0], f"neg_pairs contains node indices >= number of real nodes: max index {neg_pairs[b].max().item()}, real nodes {z.shape[0]}"

            pos_s, pos_w = self._decode_pairs(z, pos_pairs[b])
            neg_s, _ = self._decode_pairs(z, neg_pairs[b])

            all_pos_s.append(pos_s)
            all_neg_s.append(neg_s)
            if self.weighted:
                all_pos_w.append(pos_w if pos_w is not None else torch.zeros_like(pos_s))

        pos_scores = torch.stack(all_pos_s)
        neg_scores = torch.stack(all_neg_s)
        pos_w = torch.stack(all_pos_w) if self.weighted else None

        # KL: per-graph mean over nodes, averaged over ! non-empty graphs ! #BUGFIX
        nonzero_graphs = max(1, sum(1 for m in mu_list if m.shape[0] > 0))

        return pos_scores, neg_scores, pos_w, kl_total / nonzero_graphs

    def loss(self, batch, x_override=None):
        x = x_override if x_override is not None else batch['x']
        pos_scores, neg_scores, pos_w, kl = self.forward(x, batch['degree'], batch['pos_pairs'], batch['neg_pairs'], batch['edge_index_list'], batch['node_mask'], )

        recon = self.compute_loss(pos_scores, neg_scores, batch['edge_mask'], batch['neg_mask'], pos_w, batch['edge_weight'] if self.weighted else None, )
        return recon + self.kl_weight * kl

    @torch.no_grad()
    def generate( self, x, degree, edge_index=None, noise=1.0, degree_mode: str = 'directed', ):
        """
        x : [N, feat_dim]
        degree : [N, 2]

        If edge_index is provided, z is sampled from the posterior given those edges (reconstruction). If None, z is sampled from the prior.

        Decoding is streamed in row chunks. 
        """
        self.eval()
        N = x.shape[0]
        device = x.device
        chunk = self.args.gen_chunk_size

        if edge_index is not None:
            # The posterior encoder is O(E x emb_dim); reference leaves can carry tens of millions of edges. 
            # Training conditions on at most # intra_max_pairs sampled edges per leaf.
            max_ei = self.args.intra_posterior_max_edges
            if max_ei > 0 and edge_index.shape[1] > max_ei:
                keep = torch.randperm(edge_index.shape[1], device=edge_index.device)[:max_ei]
                edge_index = edge_index[:, keep]
            mu, logvar = self._encode_single(x, degree, edge_index)
            z = self._reparameterize(mu, logvar)
            logit_noise = noise
        else:
            z = torch.randn(N, self.emb_dim, device=device) * noise
            logit_noise = 0.0

        weight_fn = None
        weight_rev_fn = None
        if self.weight_mlp is not None:
            weight_fn = lambda s, e: _mlp_pair_block(self.weight_mlp, z[s:e], z)
            weight_rev_fn = lambda s, e: _mlp_pair_block_rev(self.weight_mlp, z[s:e], z)

        return streaming_degree_topk(
            N = N,
            degree = degree,
            directed = self.directed,
            weighted = self.weighted,
            degree_mode = degree_mode,
            noise = logit_noise,
            chunk_size = chunk,
            device = device,
            logits_fn = lambda s, e: (z[s:e] @ self.W) @ z.T,
            logits_rev_fn = lambda s, e: ((z @ self.W) @ z[s:e].T).T.contiguous(),
            weight_fn = weight_fn,
            weight_rev_fn = weight_rev_fn,
        )

    def init_weights(self):
        _xavier_linear_init(self)
        _xavier_init_param(self.W)
        
        self.gcn1.reset_parameters()
        self.gcn_mu.reset_parameters()
        self.gcn_var.reset_parameters()