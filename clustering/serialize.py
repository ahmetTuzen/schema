import json
import logging
import os
from typing import Optional

import torch
from torch_geometric.data import Data
from torch_geometric.utils import subgraph

logger = logging.getLogger(__name__)


def extract_inter_cluster_edges(tree: dict, leaf_labels: torch.Tensor, edge_index: torch.Tensor, edge_weight: Optional[torch.Tensor] = None) -> None:
    """
    Assign every cross-leaf edge to the tree nodes that own it.
 
    For an edge (u, v) whose endpoints lie in different leaf TreeNodes, take
    the LCA of the two leaves. The child of the LCA containing u stores
    (inside=u, outside=v); the child containing v stores (inside=v, outside=u).
    Edges whose endpoints share a leaf are skipped.
 
    Mutates 'tree' in place. Every TreeNode gets three parallel tensors:
 
        node.inter_inside_global [M] long endpoint inside this node (global id)
        node.inter_outside_global [M] long endpoint outside this node (global id)
        node.inter_attr [M] float edge weight (norm if multi-dim, 1 if None)
 
    The LCA is resolved once per distinct (leaf, leaf) pair, not per edge.
    """

    # Step 1: leaf-cluster id -> tree-node-id (only leaf) [tree node -> tree cluster in the paper, here it is excheangable..]
    lc_to_tnid = {}
    for nid, node in tree.items():
        if node.is_leaf:
            assert len(node.members) == 1, ("expected each leaf to wrap exactly one leaf cluster" )
            lc_to_tnid[node.members[0]] = nid

    # Step 2: ancestor chains for LCA.
    # ancestors[tnid] = [tnid, parent, grandparent, ..., root]
    ancestors = {}
    def chain(tnid):
        if tnid in ancestors:
            return ancestors[tnid]
        seq = []
        cur = tnid
        while cur is not None:
            seq.append(cur)
            cur = tree[cur].parent
        ancestors[tnid] = seq
        return seq

    def lca(tnid_a, tnid_b):
        """Lowest common ancestor of two tree-node ids."""
        if tnid_a == tnid_b:
            return tnid_a
        sa = set(chain(tnid_a))
        for anc in chain(tnid_b):
            if anc in sa:
                return anc
        return 0 # root fallback. safe/checked.

    # Step 3: tree-node-id of each endpoint's leaf
    src, dst = edge_index[0], edge_index[1]
    lc_src = leaf_labels[src]
    lc_dst = leaf_labels[dst]

    n_leaf_clusters = int(leaf_labels.max().item()) + 1
    lc_to_tnid_tensor = torch.full((n_leaf_clusters,), -1, dtype=torch.long)
    for lc, tnid in lc_to_tnid.items():
        lc_to_tnid_tensor[lc] = tnid

    tn_src = lc_to_tnid_tensor[lc_src]
    tn_dst = lc_to_tnid_tensor[lc_dst]

    # Step 4: keep only crossings. Intra-leaf edges are handled by intra_edge_generator.
    E = src.shape[0]
    cross = tn_src != tn_dst
    inter_count = int(cross.sum().item())
    intra_count = E - inter_count

    if edge_weight is not None and edge_weight.numel() > 0:
        w = edge_weight.float()
        if w.dim() > 1:
            w = w.norm(dim=1)
    else:
        w = torch.ones(E, dtype=torch.float)

    for nid, node in tree.items():
        node.inter_inside_global = torch.zeros(0, dtype=torch.long)
        node.inter_outside_global = torch.zeros(0, dtype=torch.long)
        node.inter_attr = torch.zeros(0, dtype=torch.float)

    if inter_count == 0:
        logger.info(f"extract_inter_cluster_edges: 0 inter edges; {intra_count} intra edges")
        return

    u = src[cross].long()
    v = dst[cross].long()
    tn_u = tn_src[cross]
    tn_v = tn_dst[cross]
    w = w[cross]
    del cross, lc_src, lc_dst, tn_src, tn_dst


    # Step 5: resolve owners. The LCA depends only on the unordered pair (tn_u, tn_v); resolve each distinct pair once and gather back onto edges.
    T = int(max(tree.keys())) + 1
    lo = torch.minimum(tn_u, tn_v)
    hi = torch.maximum(tn_u, tn_v)
    key = lo * T + hi
    uniq, inv = torch.unique(key, return_inverse=True)

    uniq_lo = (uniq // T)
    uniq_hi = (uniq % T)
    child_lo = torch.empty(uniq.numel(), dtype=torch.long)
    child_hi = torch.empty(uniq.numel(), dtype=torch.long)
    for i in range(uniq.numel()):
        a = int(uniq_lo[i])
        b = int(uniq_hi[i])
        anc = lca(a, b)
        child_lo[i] = _child_of(tree, anc, a)
        child_hi[i] = _child_of(tree, anc, b)

    # owner of the u-endpoint is the child holding tn_u, and vice versa
    u_is_lo = tn_u == uniq_lo[inv]
    owner_u = torch.where(u_is_lo, child_lo[inv], child_hi[inv])
    owner_v = torch.where(u_is_lo, child_hi[inv], child_lo[inv])
    del uniq, inv, key, lo, hi, uniq_lo, uniq_hi, child_lo, child_hi, u_is_lo, tn_u, tn_v

    # Step 6: one record per (owner, inside, outside), i.e. two per crossing
    owners = torch.cat([owner_u, owner_v])
    inside = torch.cat([u, v])
    outside = torch.cat([v, u])
    attr = torch.cat([w, w])
    del owner_u, owner_v, u, v, w

    # Step 7: group by owner via a single sort, then slice
    order = torch.argsort(owners)
    owners = owners[order]
    inside = inside[order]
    outside = outside[order]
    attr = attr[order]
    del order

    counts = torch.bincount(owners, minlength=T)
    offsets = torch.zeros(T + 1, dtype=torch.long)
    offsets[1:] = counts.cumsum(0)

    for nid in tree.keys():
        s = int(offsets[nid])
        e = int(offsets[nid + 1])
        if e > s:
            tree[nid].inter_inside_global = inside[s:e].clone()
            tree[nid].inter_outside_global = outside[s:e].clone()
            tree[nid].inter_attr = attr[s:e].clone()

    logger.info(
        f"extract_inter_cluster_edges: {inter_count} inter edges are kept as inter-cluster edges; "
        f"{intra_count} intra edges are ignored. Distinct tree-node pairs={counts.numel()}" )

def _child_of(tree, ancestor_tnid, descendant_tnid):
    """Direct child of 'ancestor_tnid' whose subtree contains 'descendant_tnid'."""
    cur = descendant_tnid
    while cur != ancestor_tnid:
        parent = tree[cur].parent
        if parent == ancestor_tnid:
            return cur
        cur = parent

    raise ValueError(f"{descendant_tnid} not in subtree of {ancestor_tnid}")

def write_pt_files(tree: dict, leaf_labels: torch.Tensor, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: Optional[torch.Tensor], out_dir: str,
                   *, dataset_name: str) -> dict:
    """
    Write one .pt per tree node and a cluster_mapping.json.

    Returns the mapping dict and saves to disk).
    """
    os.makedirs(out_dir, exist_ok=True)
    mapping = {"graphs": {}}

    # global_max_K = max children any internal node has. dataset_handler reads it to pad leaf S_pool tensors to a uniform width.
    global_max_K = 0

    for nid, node in tree.items():
        graph_id = f"g{nid}"

        # Induced subgraph of this node
        orig_idx = node.orig_idx # [N]
        local_ei, local_ew = subgraph(
            subset=orig_idx,
            edge_index=edge_index,
            edge_attr=edge_weight,
            relabel_nodes=True,
            num_nodes=leaf_labels.shape[0],
        )
        local_x = x[orig_idx]

        # Map crossing endpoints (global ids) to local row indices. Requires orig_idx to be sorted ascending.
        inside = getattr(node, 'inter_inside_global', None)
        if inside is not None and inside.numel() > 0:
            pos = torch.searchsorted(orig_idx, inside)
            pos = pos.clamp(max=orig_idx.numel() - 1)
            valid = orig_idx[pos] == inside # inside must be in this subtree
            inter_local_node = pos[valid]
            inter_external = node.inter_outside_global[valid]
            inter_edge_attr = node.inter_attr[valid]
        else:
            inter_local_node = torch.zeros(0, dtype=torch.long)
            inter_external = torch.zeros(0, dtype=torch.long)
            inter_edge_attr = torch.zeros(0, dtype=torch.float)

        # Build the Data object
        data = Data(
            x = local_x.float(),
            edge_index = local_ei.long(),
            edge_attr = local_ew.float() if local_ew is not None else None,
        )
        data.graph_id = graph_id
        data.level = node.level 

        # Parent graph id is stored in cluster_mapping.json, not in Data.
        cluster_id = _cluster_id_in_parent(tree, nid)
        data.cluster_id = cluster_id
        data.is_leaf = node.is_leaf
        data.original_node_indices = orig_idx.long()
        
        # Inter-cluster edges: local endpoint, external global endpoint, weight (if any)
        data.inter_local_node = inter_local_node.long()
        data.inter_external = inter_external.long()
        data.inter_edge_attr = inter_edge_attr.float()
        parent_gid = (f"g{node.parent}" if node.parent is not None else None)

        if node.is_leaf:
            # Parent's view: 1-D x_pool, [1,1] A_pool, [N, K_parent] S_pool
            data.x_pool = node.x_pool.float() # [feat]
            data.A_pool = node.A_pool.float() # [1, 1]
            data.S_pool = node.S_pool.float() # [N, K_parent]
            global_max_K = max(global_max_K, node.S_pool.shape[1])
        else:
            # Own clustering tensors
            data.x_pool = node.x_pool.float() # [K, feat]
            data.A_pool = node.A_pool.float() # [K, K]
            data.S_pool = node.S.float() # [N, K]
            global_max_K = max(global_max_K, node.S.shape[1])

        # Parent-view tensors if available (set by driver step 6)
        if hasattr(node, 'S_pool_parent_full'):
            data.S_pool_parent_full = node.S_pool_parent_full.float()
            data.x_pool_parent = node.x_pool_parent_slice.float()
            data.A_pool_parent = node.A_pool_parent_slice.float()
            data.parent_original_node_indices = node.parent_original_node_indices.long()
        if hasattr(node, 'S_pool_parent_slice'): # only internal
            data.S_pool_parent = node.S_pool_parent_slice.float()

        # Write
        fname = f"{dataset_name}_{graph_id}.pt"
        fpath = os.path.join(out_dir, fname)
        torch.save(data, fpath)

        # Free this node's crossings after writing for memory
        node.inter_inside_global = torch.zeros(0, dtype=torch.long)
        node.inter_outside_global = torch.zeros(0, dtype=torch.long)
        node.inter_attr = torch.zeros(0, dtype=torch.float)
        del data, local_x, local_ei, inter_local_node, inter_external, inter_edge_attr

        mapping['graphs'][graph_id] = {
            'file_path': fpath,
            'level': node.level,
            'parent': parent_gid,
            'is_leaf': node.is_leaf,
            'cluster_id': cluster_id,
        }

    mapping['_global_max_K'] = global_max_K
    json_path = os.path.join(out_dir, "cluster_mapping.json")
    with open(json_path, 'w') as f:
        json.dump(mapping, f, indent=2)

    logger.info(f"wrote {len(tree)} .pt files to {out_dir}; global_max_K={global_max_K}; mapping={json_path}")
    return mapping


def _cluster_id_in_parent(tree, nid):
    """The node's index in its parent's 'children' list. -1 for root."""
    node = tree[nid]
    if node.parent is None:
        return -1
    return tree[node.parent].children.index(nid)


def serialize(tree: dict, leaf_labels: torch.Tensor, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: Optional[torch.Tensor], out_dir: str,
            *, dataset_name: str) -> dict:
    """Extract inter-cluster edges, then write .pt files. Mutates tree."""
    extract_inter_cluster_edges(tree, leaf_labels, edge_index, edge_weight)
    return write_pt_files(tree, leaf_labels, x, edge_index, edge_weight, out_dir, dataset_name=dataset_name)