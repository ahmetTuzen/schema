import logging
from typing import Optional
 
import torch
import torch.nn.functional as F
 
logger = logging.getLogger(__name__)
 
 
def compute_soft_assignment(edge_index: torch.Tensor, labels: torch.Tensor, K: int, method: str = "propagation",
                            *,
                            n_steps: int = 3, alpha: float = 0.7) -> torch.Tensor:
    """
    Computes clustering labels into a soft assignment S [N, K].
 
    Parameters
    edge_index : [2, E] LOCAL edges of this node's induced subgraph (0..N-1).
    x : [N, feat] node features (used only by method="gnn").
    labels : [N] hard child label per node, values in 0 to K-1.
    K : number of children (columns of S).
    n_steps : propagation iterations (propagation only).
    alpha : propagation mixing coefficient in [0,1], Higher = membership spreads more under graph propagation
    gnn_model : a pre-trained SoftAssignGNN (gnn only). Must output K-dim logits.
 
    Returns
    S : [N, K] rows sum to 1, non-negative.
    """
    if method == "propagation":
        return _soft_assignment_propagation(edge_index, labels, K, n_steps=n_steps, alpha=alpha)
    else:
        raise ValueError(f"Unknown soft-assignment method: {method}")
 
 
def _soft_assignment_propagation(edge_index: torch.Tensor,
                               labels: torch.Tensor,
                               K: int,
                               n_steps: int = 3,
                               alpha: float = 0.7) -> torch.Tensor:
    """
    Equation (1)
    All operations are sparse scatter-adds over the edge list, no dense operations [N, N].
    """
    N = labels.shape[0]
    device = labels.device
 
    S = F.one_hot(labels, num_classes=K).float() # [N, K]
    S_init = S.clone()
 
    # symmetrize + add self-loops, build normalized edge weights
    src, dst = edge_index[0], edge_index[1]
    self_idx = torch.arange(N, device=device)
    si = torch.cat([src, dst, self_idx]) # symmetric + self
    di = torch.cat([dst, src, self_idx])
 
    deg = torch.zeros(N, device=device)
    deg.scatter_add_(0, si, torch.ones(si.shape[0], device=device))
    dinv_sqrt = deg.clamp(min=1.0).pow(-0.5)
    norm = dinv_sqrt[si] * dinv_sqrt[di] # [E_sym] edge weights
 
    # sparse normalized adjacency, built once: A[d, s] = norm for each (si, di) pair
    A = torch.sparse_coo_tensor(torch.stack([di, si]), norm, (N, N)).coalesce()

    for _ in range(n_steps):
        agg = torch.sparse.mm(A, S)
        S = alpha * agg + (1.0 - alpha) * S_init
        S = S / S.sum(dim=1, keepdim=True).clamp(min=1e-12)
 
    return S

 
 
def compute_x_pool(S: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    x_pool[c] = sum_n S[n, c] * x[n] -> [K, feat]
 
    Soft aggregation of node features per child using membership weights.
    Each child representation is the weighted sum of node features, 
    where weights are given by the soft assignment matrix S.
    """
    return S.t() @ x # [K, feat]
 
 
def compute_A_pool(S: torch.Tensor, edge_index: torch.Tensor, edge_weight: Optional[torch.Tensor] = None,
                   *,
                   mode: str = "auto", chunk_size: int = 2_000_000) -> torch.Tensor:
    """
    Computes A_pool = S^T A S using either:
      - sparse matmul (recommended for large graphs)
      - einsum (fallback / small graphs)
      - chunked einsum (memory)

    mode:
        "auto" 
        "sparse" -> force sparse.mm
        "einsum" -> force full einsum
        "chunked" -> force chunked einsum
    """

    K = S.shape[1]
    src, dst = edge_index[0], edge_index[1]
    E = src.shape[0]

    if edge_weight is None or edge_weight.numel() == 0:
        edge_weight = torch.ones(E, device=S.device)

    # 1) AUTO routing logic
    if mode == "auto":
        if E > 250_000:
            mode = "sparse"
        else:
            mode = "einsum"

    # 2) Sparse matrix multiplication path
    if mode == "sparse":
        A = torch.sparse_coo_tensor(torch.stack([src, dst]), edge_weight, size=(S.shape[0], S.shape[0])).coalesce()
        AS = torch.sparse.mm(A, S)
        return S.t() @ AS

    # 3) Chunked einsum (memory-safe)
    if mode == "chunked":
        A_pool = torch.zeros(K, K, device=S.device)
        for start in range(0, E, chunk_size):
            end = min(start + chunk_size, E)

            si = src[start:end]
            di = dst[start:end]
            w = edge_weight[start:end]

            A_pool += torch.einsum('ek,e,em->km', S[si], w, S[di])

        return A_pool

    # 4) Full einsum path (small graphs only)
    
    return torch.einsum('ek,e,em->km', S[src], edge_weight, S[dst])
 
 
def compute_node_artifacts(edge_index: torch.Tensor, x: torch.Tensor, labels: torch.Tensor, K: int,
                           *,
                           edge_weight: Optional[torch.Tensor] = None, n_steps: int = 3, alpha: float = 0.7):
    """
    Compute (S, x_pool, A_pool)
 
    edge_index : LOCAL induced edges (0..N-1).
    x : [N, feat] local node features.
    labels : [N] hard child labels 0..K-1.
    K : number of children.
 
    Returns (S [N,K], x_pool [K,feat], A_pool [K,K]).
    """
    S = compute_soft_assignment(edge_index, labels, K, n_steps=n_steps, alpha=alpha)
    
    x_pool = compute_x_pool(S, x)
    A_pool = compute_A_pool(S, edge_index, edge_weight)

    return S, x_pool, A_pool