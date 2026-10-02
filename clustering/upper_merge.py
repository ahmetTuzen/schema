import logging
from dataclasses import dataclass, field
from typing import Optional

import igraph as ig
import leidenalg
import torch

from supernode import SupernodeGraph

logger = logging.getLogger(__name__)


@dataclass
class TreeNode:
    node_id: int
    parent: Optional[int]
    children: list = field(default_factory=list) # list[int] of child node_ids
    members: list = field(default_factory=list) # list[int] of leaf cluster ids

    is_leaf: bool = False # True if a Leiden leaf cluster
    level: int = 0 # 0 = root, just for reporting purposes.


class _IdGen:
    """Sequential id source. Root = 0."""
    def __init__(self):
        self.n = 0
        
    def next(self) -> int:
        i = self.n
        self.n += 1
        return i


def _check_children_cover(tree, parent_id):
    """Every member of parent must appear in exactly one child, and children must be non-empty."""
    node = tree[parent_id]
    assert node.children, f"node {parent_id}: internal node has no children"

    covered = []
    for cid in node.children:
        covered.extend(tree[cid].members)

    missing = set(node.members) - set(covered)
    assert not missing, (f"node {parent_id}: {len(missing)} members not covered by children, e.g. {sorted(missing)[:5]}")
    assert len(covered) == len(set(covered)), f"node {parent_id}: a member appears in multiple children"
    assert len(covered) == len(node.members), (f"node {parent_id}: children cover {len(covered)} members but parent has {len(node.members)}")


def build_upper_tree(sg: SupernodeGraph, stats: dict, branching_cap: int = 16, seed: int = 0) -> dict:
    """
    Returns dict[int, TreeNode], rooted at id 0.

    stats : the dict returned by supernode.report()
    """
    K = sg.num_nodes # number of supernodes = number of leaf clusters
    isolated_frac = stats["isolated"] / max(K, 1)

    if K <= branching_cap:
        logger.info(f"upper_merge: Leaf clusters are less than cap, flat (K={K} <= b={branching_cap})")
        return _build_flat(sg, K)

    logger.info(f"upper_merge: Recursive Leiden called with (K={K}, b={branching_cap})")
    return _build_recursive_leiden(sg, branching_cap, seed)


def _build_flat(sg: SupernodeGraph, K: int) -> dict:
    """
    No hierarchy beyond the leaf partition.
    """
    tree = {}
    ids = _IdGen()

    root_id = ids.next()
    tree[root_id] = TreeNode(
        node_id=root_id,
        parent=None,
        level=0,
        is_leaf=False,
        members=list(range(K)),
    )

    for leaf_cluster_id in range(K):
        child_id = ids.next()
        tree[root_id].children.append(child_id)
        tree[child_id] = TreeNode(
            node_id=child_id,
            parent=root_id,
            level=1,
            is_leaf=True,
            members=[leaf_cluster_id],
        )

    _check_children_cover(tree, root_id)
    return tree


def _build_recursive_leiden(sg: SupernodeGraph, b: int, seed: int) -> dict:
    tree = {}
    ids = _IdGen()

    root_id = ids.next()
    tree[root_id] = TreeNode(
        node_id=root_id,
        parent=None,
        level=0,
        is_leaf=False,
        members=list(range(sg.num_nodes)),
    )

    _split_into_node(
        tree=tree,
        ids=ids,
        parent_id=root_id,
        parent_level=0,
        supernode_ids=list(range(sg.num_nodes)),
        sg=sg,
        b=b,
        seed=seed,
    )

    return tree


def _split_into_node(tree, ids, parent_id, parent_level, supernode_ids, sg, b, seed):
    """
    Populate tree[parent_id].children by partitioning supernode_ids.

    Cses:
      1. len(supernode_ids) <= b: attach all as direct leaf children.
      2. Induced subgraph has no edges: no community structure -> size-balanced chunking.
      3. Leiden returns 1 group: no community structure -> size-balanced chunking.

    Recursive case: Leiden splits into >1 groups; each group becomes either
    a direct leaf (singleton group) or an internal node (multi-supernode group).

    """
    n = len(supernode_ids)
    assert n > 1, "should not be called with a single supernode"

    # Base case 1: small enough -> flat attach
    if n <= b:
        _attach_as_direct_leaves(tree, ids, parent_id, parent_level, supernode_ids, reason=f"n={n} <= b={b}")
        _check_children_cover(tree, parent_id)
        return

    # Build induced subgraph for this scope
    sub_ei, sub_w = _induce_supernode_subgraph(sg, supernode_ids)

    # Base case 2: no edges -> size-balanced chunking
    if sub_ei.shape[1] == 0:
        _chunk_and_attach(tree, ids, parent_id, parent_level, supernode_ids, sg=sg, b=b, seed=seed, reason="no edges in scope")
        _check_children_cover(tree, parent_id)
        return

    # Leiden
    labels = _leiden_on_supernode_subgraph(sub_ei, sub_w, n, seed)
    K_groups = int(labels.max().item()) + 1

    # Base case 3: stall -> size-balanced chunking
    if K_groups == 1:
        _chunk_and_attach(tree, ids, parent_id, parent_level, supernode_ids, sg=sg, b=b, seed=seed, reason="Leiden returned 1 group")
        _check_children_cover(tree, parent_id)
        return

    # Map labels back to supernode groupings
    
    labels_list = labels.tolist()
    buckets_by_label = [[] for _ in range(K_groups)]
    for i, k in enumerate(labels_list):
        buckets_by_label[k].append(supernode_ids[i])
    groups = [g for g in buckets_by_label if g]

    n_leiden = len(groups)

    # Enforce the cap: treat each Leiden group as an atomic unit, bucket them by total size.
    if n_leiden > b:
        sizes = [sum(int(sg.sizes[s]) for s in g) for g in groups]
        order = sorted(range(n_leiden), key=lambda i: (-sizes[i], i))
        buckets = [[] for _ in range(b)]
        loads = [0] * b
        for i in order:
            j = min(range(b), key=lambda t: (loads[t], t))
            buckets[j].extend(groups[i])
            loads[j] += sizes[i]
        groups = [bk for bk in buckets if bk]
        logger.info(f"upper_merge: Level {parent_level} node {parent_id}: Leiden returned {n_leiden} groups (cap={b}); regrouped into {len(groups)} buckets.")
    else:
        logger.info(f"upper_merge: Level {parent_level} node {parent_id}: {n_leiden} groups (Leiden split); sizes={[len(g) for g in groups]}")

    _attach_groups(tree, ids, parent_id, parent_level, groups, sg=sg, b=b, seed=seed)
    _check_children_cover(tree, parent_id)


def _chunk_into_groups(supernode_ids, sg, b):
    """
    Split supernode_ids into <= b size-balanced groups (largest-first, least-loaded).
    Used when the induced supernode subgraph carries no usable community structure.
    """
    order = sorted(supernode_ids, key=lambda s: (-int(sg.sizes[s]), s))
    groups = [[] for _ in range(b)]
    loads = [0] * b
    for sid in order:
        j = min(range(b), key=lambda i: (loads[i], i))
        groups[j].append(sid)
        loads[j] += int(sg.sizes[sid])
    return [g for g in groups if g]


def _chunk_and_attach(tree, ids, parent_id, parent_level, supernode_ids, sg, b, seed, reason):
    """Size-balanced fallback split, then attach the resulting groups."""
    n = len(supernode_ids)
    groups = _chunk_into_groups(supernode_ids, sg, b)

    if len(groups) <= 1 or max(len(g) for g in groups) >= n:
        logger.warning(f"upper_merge: Level {parent_level} node {parent_id}: chunking made no progress on {n} supernodes; attaching all as direct leaves.")

        _attach_as_direct_leaves(tree, ids, parent_id, parent_level, supernode_ids, reason=reason)
        return

    logger.info(f"upper_merge: Level {parent_level} node {parent_id}: no community structure ({reason}); "
                f"chunking {n} supernodes into {len(groups)} balanced groups; sizes={[len(g) for g in groups]}")
    

    _attach_groups(tree, ids, parent_id, parent_level, groups, sg=sg, b=b, seed=seed)


def _attach_groups(tree, ids, parent_id, parent_level, groups, sg, b, seed):
    """
    Attach a list of supernode groups as children of parent_id.
    Singleton groups become leaves; multi-supernode groups become internal nodes and recurse.
    """
    child_level = parent_level + 1
    for group in groups:
        if len(group) == 0:
            continue
        if len(group) == 1:
            _attach_leaf(tree, ids, parent_id, child_level, group[0])
            continue

        internal_id = ids.next()
        tree[parent_id].children.append(internal_id)
        tree[internal_id] = TreeNode(
            node_id=internal_id, parent=parent_id, level=child_level,
            is_leaf=False, members=list(group),
        )
        _split_into_node(
            tree=tree, ids=ids,
            parent_id=internal_id, parent_level=child_level,
            supernode_ids=list(group),
            sg=sg, b=b, seed=seed,
        )


def _attach_as_direct_leaves(tree, ids, parent_id, parent_level, supernode_ids, reason):
    """
    Attach every supernode_id as a direct leaf child of parent_id.
    No intermediate internal node is created. Caller must ensure len(supernode_ids) <= b,
    or accept that the branching cap is exceeded here.
    """
    logger.info(f"upper_merge: Level {parent_level} node {parent_id}: attaching {len(supernode_ids)} supernodes as direct leaves.\tReason: ({reason})")
    
    child_level = parent_level + 1
    for sid in supernode_ids:
        _attach_leaf(tree, ids, parent_id, child_level, sid)


def _attach_leaf(tree, ids, parent_id, level, leaf_cluster_id):
    """Create a single leaf TreeNode under parent_id."""
    leaf_id = ids.next()
    tree[parent_id].children.append(leaf_id)
    tree[leaf_id] = TreeNode(
        node_id = leaf_id,
        parent = parent_id,
        level = level,
        is_leaf = True,
        members = [leaf_cluster_id],
    )


def _induce_supernode_subgraph(sg: SupernodeGraph, supernode_ids):
    """
    Return (edge_index, edge_weight) of the supernode graph restricted to
    'supernode_ids'. Re-indexed 0..len(supernode_ids)-1.
    """
    device = sg.edge_index.device

    src = sg.edge_index[0]
    dst = sg.edge_index[1]
    w = sg.edge_weight

    keep = torch.zeros(sg.num_nodes, dtype=torch.bool, device=device)
    keep[torch.tensor(supernode_ids, device=device)] = True

    mask = keep[src] & keep[dst]

    src = src[mask]
    dst = dst[mask]
    w = w[mask]

    remap = torch.full((sg.num_nodes,), -1, dtype=torch.long, device=device)
    remap[torch.tensor(supernode_ids, device=device)] = torch.arange(len(supernode_ids), device=device)

    new_src = remap[src]
    new_dst = remap[dst]

    if new_src.numel() == 0:
        return (
            torch.zeros(2, 0, dtype=torch.long, device=device),
            torch.zeros(0, dtype=torch.float, device=device),
        )

    return torch.stack([new_src, new_dst]), w


def _leiden_on_supernode_subgraph(edge_index, edge_weight, n, seed):
    """Leiden (modularity) on a small supernode subgraph. Returns labels [n]."""
    edges = list(zip(edge_index[0].tolist(), edge_index[1].tolist()))
    g = ig.Graph(n=n, edges=edges, directed=False)
    g.es["weight"] = edge_weight.tolist()
    g.simplify(combine_edges={"weight": "sum"})
    part = leidenalg.find_partition(g, leidenalg.ModularityVertexPartition, weights=g.es["weight"], seed=seed)
    return torch.tensor(part.membership, dtype=torch.long)