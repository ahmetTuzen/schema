import logging
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)


@dataclass
class SupernodeGraph:
    edge_index: torch.Tensor # [2, E_super] long, undirected (lo, hi)
    edge_weight: torch.Tensor # [E_super] float
    num_nodes: int # = K (number of leaf clusters)
    sizes: torch.Tensor # [K] long, node count per leaf cluster

def build(edge_index: torch.Tensor,
          labels: torch.Tensor,
          K: int,
          edge_weight: torch.Tensor = None
        ) -> SupernodeGraph:
    """
    Vectorized construction of the supernode graph.

    edge_index : [2, E] node-level edges. 
    labels : [N] cluster id per node.
    K : number of clusters = labels.max() + 1.
    edge_weight : [E] optional (inherited from reference graph).
    """
    device = edge_index.device

    src, dst = edge_index[0], edge_index[1]
    csrc, cdst = labels[src], labels[dst]

    cross = csrc != cdst
    csrc, cdst = csrc[cross], cdst[cross]

    if edge_weight is not None:
        w = edge_weight[cross]
    else:
        w = torch.ones(int(cross.sum()), dtype=torch.float, device=device)

    # canonicalize undirected pair: lo <= hi
    lo = torch.minimum(csrc, cdst).long()
    hi = torch.maximum(csrc, cdst).long()

    # integer key per unique pair
    key = lo * K + hi
    uniq, inv = torch.unique(key, return_inverse=True)

    super_w = torch.zeros(uniq.shape[0], dtype=w.dtype, device=device)
    super_w.scatter_add_(0, inv, w)
    
    lo_u = uniq // K
    hi_u = uniq %  K
    super_ei = torch.stack([lo_u, hi_u], dim=0)

    logger.info(f"supernode graph: {K} nodes, {super_ei.shape[1]} undirected edges")
    return SupernodeGraph(edge_index=super_ei, edge_weight=super_w, num_nodes=K, sizes=torch.bincount(labels, minlength=K))


def report(sg: SupernodeGraph) -> dict:
    """
    Returns a dict and logs a summary.
    """
    device = sg.edge_index.device

    K = sg.num_nodes
    E = sg.edge_index.shape[1]
    max_E = K * (K - 1) // 2

    deg = torch.zeros(K, dtype=torch.long, device=device)
    deg.scatter_add_(0, sg.edge_index[0], torch.ones(E, dtype=torch.long, device=device))
    deg.scatter_add_(0, sg.edge_index[1], torch.ones(E, dtype=torch.long, device=device))

    stats = {
        "supernodes": K,
        "supernode_edges": E,
        "max_possible_edges": max_E,
        "density": E / max_E if max_E > 0 else 0.0,
        "deg_min": int(deg.min()),
        "deg_max": int(deg.max()),
        "deg_mean": float(deg.float().mean()),
        "isolated": int((deg == 0).sum()),
        "weight_min": float(sg.edge_weight.min()),
        "weight_max": float(sg.edge_weight.max()),
        "weight_mean": float(sg.edge_weight.mean()),
    }

    logger.info(
        f"supernode report: K={K} edges={E}/{max_E} "
        f"density={stats['density']:.6f} "
        f"deg min/mean/max={stats['deg_min']}/{stats['deg_mean']:.1f}/"
        f"{stats['deg_max']} isolated={stats['isolated']}"
    )
    return stats