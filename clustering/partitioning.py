import logging
from dataclasses import dataclass

import igraph as ig
import leidenalg
import torch

logger = logging.getLogger(__name__)


@dataclass
class PartitionResult:
    labels: torch.Tensor # [num_nodes] long
    K: int # number of clusters
    modularity: float
    method: str # "leiden" or "louvain" (mostly leiden)


def _build_igraph(edge_index, num_nodes, edge_weight):
    """ 
    Build undirected, simplified igraph, for algorithms to work
    
    Returns (graph, weights_or_None).
    """
    CHUNK = 5_000_000 # implementing chunk by chunk for memory enhancement

    g = ig.Graph(n=num_nodes, directed=False)
    src, dst = edge_index[0], edge_index[1]
    for i in range(0, src.numel(), CHUNK):
        g.add_edges(list(zip(src[i:i + CHUNK].tolist(), dst[i:i + CHUNK].tolist())))

    if edge_weight is not None:
        g.es["weight"] = edge_weight.tolist()
        g.simplify(combine_edges={"weight": "sum"})
        return g, g.es["weight"]

    g.simplify()

    return g, None


def _find_partition(g, objective, seed, weights=None):
    """Single leidenalg call. Shared by the flat partition and the splitter."""
    if objective == "modularity":
        return leidenalg.find_partition(
            g, leidenalg.ModularityVertexPartition,
            weights=weights,
            seed=seed,
        )
    else:
        raise ValueError(f"Unknown Leiden objective: {objective}")


def _split_oversized(g, nodes, cap, seed, objective, keep_frac, depth, max_depth):
    """Splits oversized communities (args.max_leaf_size)."""

    if nodes.numel() <= cap:
        return [nodes]
    if depth > max_depth:
        logger.warning(f"max_leaf_size: depth limit {max_depth} reached with a cluster of {nodes.numel()} nodes")
        return [nodes]

    nodes = torch.sort(nodes).values
    sub = g.subgraph(nodes.tolist())
    sub_w = sub.es["weight"] if "weight" in sub.es.attributes() else None

    part = _find_partition(sub, objective, seed, weights=sub_w)
    labels = torch.tensor(part.membership, dtype=torch.long)
    K = int(labels.max().item()) + 1

    if K == 1 or int(torch.bincount(labels, minlength=K).max()) >= keep_frac * nodes.numel():
        logger.warning(f"max_leaf_size: no split found for a cluster of {nodes.numel()} nodes; keeping it whole")
        return [nodes]

    out = []
    for c in range(K):
        child = nodes[labels == c]
        if child.numel() > cap:
            logger.info(f"max_leaf_size: depth {depth}: {nodes.numel()} -> child {child.numel()}, recursing")
            out.extend(_split_oversized(g, child, cap, seed, objective, keep_frac, depth + 1, max_depth))
        else:
            out.append(child)
    return out


def enforce_max_leaf_size(g,
                          labels: torch.Tensor,
                          cap: int,
                          seed: int = 0,
                          objective: str = "modularity",
                          keep_frac: float = 0.95,
                          max_depth: int = 8) -> torch.Tensor:
    """
    Split any cluster larger than 'cap' by re-running community detection on its induced subgraph, recursively. 
    """
    if cap is None or cap <= 0:
        return labels

    K = int(labels.max().item()) + 1
    counts = torch.bincount(labels, minlength=K)
    if int(counts.max()) <= cap:
        logger.info(f"max_leaf_size={cap}: largest cluster is {int(counts.max())}, nothing to split")
        return labels

    pieces = []
    for c in range(K):
        nodes = (labels == c).nonzero(as_tuple=True)[0]
        if nodes.numel() == 0:
            continue
        if nodes.numel() <= cap:
            pieces.append(nodes)
        else:
            logger.info(f"max_leaf_size={cap}: splitting cluster {c} with {nodes.numel()} nodes")
            pieces.extend(_split_oversized(g, nodes, cap, seed, objective, keep_frac, depth=1, max_depth=max_depth, ))

    new_labels = torch.empty_like(labels)
    for i, nodes in enumerate(pieces):
        new_labels[nodes] = i

    logger.info(
        f"max_leaf_size={cap}: {K} -> {len(pieces)} clusters, "
        f"largest {int(torch.bincount(new_labels, minlength=len(pieces)).max())}"
    )
    return new_labels


def partition(method: str,
              edge_index: torch.Tensor,
              num_nodes: int,
              edge_weight = None,
              objective: str = "modularity",
              seed: int = 0,
              max_leaf_size: int = 0,
              ) -> PartitionResult:

    method = method.lower()
    g, weights = _build_igraph(edge_index, num_nodes, edge_weight)

    if method == "louvain":
        part = g.community_multilevel(weights=weights)

    elif method == "leiden":
        part = _find_partition(g, objective, seed, weights=weights)

    else:
        raise ValueError(f"Unknown clustering method: {method}")

    labels = torch.tensor(part.membership, dtype=torch.long)
    K = int(labels.max().item()) + 1
    mod = float(part.modularity)

    logger.info(f"{method} partition: {K} clusters, modularity={mod:.4f}")

    # when max_leaf_size <= 0 (the default) do not split
    if max_leaf_size and max_leaf_size > 0:
        labels = enforce_max_leaf_size(
            g, labels, cap=max_leaf_size, seed=seed,
            objective=(objective if method == "leiden" else "modularity"),
            weights=weights,
        )
        K = int(labels.max().item()) + 1
        
        mod = float(g.modularity(labels.tolist(), weights=weights))
        logger.info(f"{method} partition after max_leaf_size: {K} clusters, modularity={mod:.4f}")

    return PartitionResult(labels=labels, K=K, modularity=mod, method=method)