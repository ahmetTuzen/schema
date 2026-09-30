import logging
from typing import Optional

import torch
from torch_geometric.utils import subgraph

from pooling import compute_node_artifacts

logger = logging.getLogger(__name__)


def compute_all_artifacts(tree: dict, leaf_labels: torch.Tensor, edge_index: torch.Tensor, x: torch.Tensor,
                          *,
                          edge_weight: Optional[torch.Tensor] = None, n_steps: int = 3, alpha: float = 0.7, gnn_model=None) -> None:
    """
    Populate every internal TreeNode with S, x_pool, A_pool, for generation phases to use, maybe some additional fields for future uses.

    Parameters
    ----------
    tree : dict[int, TreeNode] from upper_merge.build_upper_tree.
    leaf_labels : [num_nodes] leaf-cluster id per ORIGINAL node (from partition).
    edge_index : [2, E] ORIGINAL graph edges (global node ids).
    x : [num_nodes, feat] ORIGINAL node features.
    edge_weight : [E] optional.
    method : "propagation".

    After this call, each TreeNode has (added attributes):
        internal cluster:
            .S [N_X, K] soft assignment over children
            .x_pool [K, feat] pooled features per child
            .A_pool [K, K] pooled adjacency per child
            .orig_idx [N_X] original node ids in this subtree
            .child_labels [N_X] which direct child each node falls under (0..K-1)
            .local_edge_index [2,E_X] induced edges, re-indexed 0..N_X-1

        leaf clusters (filled from parent):
            .x_pool [feat] the parent's x_pool row for this leaf (1-D!)
            .A_pool [1, 1] the parent's A_pool self-density for this leaf
            .orig_idx [N_leaf] original node ids in this leaf cluster
    """
    
    internal_ids = sorted((nid for nid, n in tree.items() if not n.is_leaf), key=lambda nid: tree[nid].level)

    for nid in internal_ids:
        node = tree[nid]
        children = node.children
        K = len(children)

        # 1. original nodes in this subtree = union of leaf clusters in members
        members_t = torch.tensor(node.members, dtype=torch.long)
        node_mask = torch.isin(leaf_labels, members_t) # [num_nodes] bool
        orig_idx = node_mask.nonzero(as_tuple=True)[0] # [N_X] global ids
        N_X = orig_idx.shape[0]

        # 2. child label per node: which DIRECT child does each node connect.
        # Build a map: leaf-cluster id -> child index (0..K-1).
        # A child may be a subtree (covers many leaf clusters) 
        # or a single leaf cluster; 
        # either way child.members lists its leaf clusters.
        leafclus_to_child = {}
        for child_idx, child_nid in enumerate(children):
            for lc in tree[child_nid].members:
                leafclus_to_child[lc] = child_idx

        # map each original node (via its leaf cluster) to its child index
        node_leafclus = leaf_labels[orig_idx] # [N_X]
        child_labels = torch.tensor(
            [leafclus_to_child[int(lc)] for lc in node_leafclus],
            dtype=torch.long,
        ) # [N_X] in 0..K-1

        # 3. induced subgraph, re-indexed to 0..N_X-1
        local_ei, local_ew = subgraph(
            subset=orig_idx,
            edge_index=edge_index,
            edge_attr=edge_weight,
            relabel_nodes=True,
            num_nodes=leaf_labels.shape[0],
        )
        local_x = x[orig_idx] # [N_X, feat]

        # 4. compute artifacts
        S, x_pool, A_pool = compute_node_artifacts(
            edge_index = local_ei,
            x = local_x,
            labels = child_labels,
            K = K,
            edge_weight= local_ew,
            n_steps = n_steps,
            alpha = alpha,
        )

        # 5. attach to the internal node
        node.S = S
        node.x_pool = x_pool # [K, feat]
        node.A_pool = A_pool # [K, K]
        node.orig_idx = orig_idx
        node.child_labels = child_labels
        node.local_edge_index = local_ei

        # 6. hand each child its slice (parent's view)
        # Rows of parent's S aligned with orig_idx are in ascending global-id # order; 
        # the child later re-extracts its nodes via isin, 
        # so slicing rows by child_labels==child_idx preserves the child's order.
        for child_idx, child_nid in enumerate(children):
            child = tree[child_nid]
            child_mask = (child_labels == child_idx)
            child.x_pool = x_pool[child_idx] # 1-D [feat]
            child.A_pool = A_pool[child_idx, child_idx].view(1, 1)
            child.S_pool = S[child_mask] # [N_child, K_parent]

            # parent-view (inter-edge source role) leaf and internal:
            child.S_pool_parent_full = S # [N_parent, K_parent]
            child.x_pool_parent_slice = x_pool # [K_parent, feat]
            child.A_pool_parent_slice = A_pool # [K_parent, K_parent]
            child.parent_original_node_indices = orig_idx # [N_parent]
            if not tree[child_nid].is_leaf:
                child.S_pool_parent_slice = S[child_mask] # [N_child, K_parent]
                
        if N_X >= 1000 or node.level <= 2:
            logger.info(f"node {nid} (level {node.level}): N={N_X}, K={K}, S={tuple(S.shape)}, x_pool={tuple(x_pool.shape)}, A_pool={tuple(A_pool.shape)}")
            
    # Every leaf has a parent (root covers all), so all leaves got their x_pool/A_pool in step 6 above. 
    # Fill orig_idx for leaves too:
    for nid, node in tree.items():
        if node.is_leaf:
            members_t = torch.tensor(node.members, dtype=torch.long)
            node_mask = torch.isin(leaf_labels, members_t)
            node.orig_idx = node_mask.nonzero(as_tuple=True)[0]