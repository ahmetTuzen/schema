import os
import tempfile
import logging
from typing import Optional
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.data import Data

logger = logging.getLogger(__name__)


def _infer_directed(model, default: bool = True) -> bool:
    """Pull directedness from a model's config."""
    if model is None:
        return default
    return bool(getattr(model, 'directed', default))


def _degree_mode_for(model) -> str:
    """Map model.directed to the degree_mode string the intra generator uses."""
    return 'directed' if _infer_directed(model, default=True) else 'undirected'


def _load_subgraph(path):
    return torch.load(path, weights_only=False)


class _BoundedCache:
    """Small LRU cache for parent original_node_indices, reused by siblings."""

    def __init__(self, maxsize: int = 16):
        self.maxsize = maxsize
        self._d = OrderedDict()

    def get(self, key):
        if key in self._d:
            self._d.move_to_end(key)
            return self._d[key]
        return None

    def put(self, key, value):
        self._d[key] = value
        self._d.move_to_end(key)
        if len(self._d) > self.maxsize:
            self._d.popitem(last=False)


def _parent_orig_idx(data, parent_file_cache: dict, parent_path_map: dict, gid: str) -> Optional[torch.Tensor]:
    """
    Return parent's original_node_indices as a tensor, or None for root.
    """
    if hasattr(data, 'parent_original_node_indices') and data.parent_original_node_indices is not None:
        return data.parent_original_node_indices.cpu().long()

    parent_path = parent_path_map.get(gid, None)
    if parent_path is None:
        return None # root
    if parent_path in parent_file_cache:
        return parent_file_cache[parent_path]

    pdata = _load_subgraph(parent_path)
    pi = pdata.original_node_indices.cpu().long()
    parent_file_cache[parent_path] = pi
    return pi


def scan_leaf_meta(dataset) -> tuple:
    """
    Metadata pre-pass over the leaf dataset, aim is the original_node_indices of each leaf, and the total number of nodes.
    So we can evaluate our method on memorization.
    """
    leaf_meta = {}
    total = 0
    max_idx = -1

    for path in dataset.file_paths:
        leaf = _load_subgraph(path)
        orig_idx = leaf.original_node_indices.cpu().long()
        leaf_meta[leaf.graph_id] = {
            'original_idx': orig_idx,
            'N': int(orig_idx.shape[0]),
            'level': leaf.level,
            'cluster_id': leaf.cluster_id,
            'file_path': path,
        }
        total += int(orig_idx.shape[0])
        m = int(orig_idx.max().item()) if orig_idx.numel() > 0 else -1
        max_idx = max(max_idx, m)
        del leaf

    num_nodes = max_idx + 1
    assert total == num_nodes, (f"Leaves do not partition the node set: sum of leaf sizes {total} != max_id+1 {num_nodes}. Memmap scatter assembly requires a partition." )
    logger.info(f"[assembler] pre-pass: {len(leaf_meta)} leaves, N={num_nodes}")
    return leaf_meta, num_nodes


def assemble_leaves_streaming(
        node_generator,
        intra_edge_model,
        dataloader,
        x_mean: torch.Tensor,
        x_std: torch.Tensor,
        leaf_meta: dict,
        x_mm: np.memmap,
        device: str = 'cpu',
        node_noise: float = 0.0,
        intra_noise: float = 0.0,
        weighted: bool = False,
    ) -> tuple:
    """
    Generate x and intra edges for all leaves, streaming to disk-backed memmap.
    """

    x_mean = x_mean.to(device)
    x_std = x_std.to(device)

    dataset = dataloader.dataset
    node_generator.eval()
    if intra_edge_model is not None:
        intra_edge_model.eval()
        degree_mode = _degree_mode_for(intra_edge_model)

    local_loader = DataLoader(dataset, batch_size = dataloader.batch_size, shuffle = False, collate_fn = dataloader.collate_fn, num_workers = 0,)
    
    edge_src_parts, edge_dst_parts, edge_w_parts = [], [], []

    ptr = 0 # running index into dataset.file_paths

    with torch.no_grad():
        for batch in local_loader:
            x_pool = batch['x_pool'].to(device)
            S = batch['S'].to(device)
            mask = batch['mask'].to(device)

            x_gen = node_generator.generate(x_pool, S, mask=mask, noise=node_noise)

            # Inverse transform per feature type. 
            # Binary: the model emits logits; sample from the sigmoid
            # Continuous: undo normalization.
            feature_type = getattr(getattr(node_generator, 'args', None), 'feature_type', 'continuous')
            if feature_type == 'binary':
                probs = torch.sigmoid(x_gen)
                if node_noise > 0:
                    x_gen = torch.bernoulli(probs)
                else:
                    F_dim = probs.shape[-1]
                    k = probs.sum(dim=-1).round().long().clamp(min=0, max=F_dim)
                    x_gen = torch.zeros_like(probs)
                    kmax = int(k.max().item()) if k.numel() else 0
                    if kmax > 0:
                        idx = probs.topk(kmax, dim=-1).indices # [..., kmax]
                        keep = torch.arange(kmax, device=probs.device) < k.unsqueeze(-1)
                        x_gen.scatter_(-1, idx, keep.to(probs.dtype))
            else:
                x_gen = x_gen * x_std + x_mean

            B = x_pool.shape[0]
            for i in range(B):
                if ptr >= len(dataset):
                    break

                N_real = int(mask[i].sum().item())
                x_i = x_gen[i, :N_real]

                leaf = _load_subgraph(dataset.file_paths[ptr])
                ptr += 1
                gid = leaf.graph_id
                orig_idx = leaf_meta[gid]['original_idx']

                assert len(orig_idx) == N_real, f"mismatch at {gid}: orig_idx has {len(orig_idx)}, mask says N={N_real}"

                # scatter features to disk, global order, no RAM copy kept
                x_mm[orig_idx.numpy()] = x_i.cpu().numpy()

                # intra edges for this leaf  
                if intra_edge_model is not None:
                    src, dst = leaf.edge_index[0], leaf.edge_index[1]
                    in_degree = torch.zeros(N_real, dtype=torch.float)
                    out_degree = torch.zeros(N_real, dtype=torch.float)
                    in_degree.scatter_add_( 0, dst, torch.ones(dst.shape[0]))
                    out_degree.scatter_add_(0, src, torch.ones(src.shape[0]))
                    degree = torch.stack([in_degree, out_degree], dim=1).to(device)

                    ei, ew = intra_edge_model.generate(
                        x = x_i.to(device),
                        degree = degree,
                        edge_index = leaf.edge_index.to(device) if intra_noise == 0.0 else None,
                        noise = intra_noise,
                        degree_mode = degree_mode,
                    )

                    if ei is not None and ei.shape[1] > 0:
                        ei = ei.cpu().long()
                        # vectorized local -> global ids
                        edge_src_parts.append(orig_idx[ei[0]])
                        edge_dst_parts.append(orig_idx[ei[1]])
                        if weighted and ew is not None:
                            edge_w_parts.append(ew.cpu())

                del leaf, x_i

    assert ptr == len(dataset)
    x_mm.flush()

    return edge_src_parts, edge_dst_parts, edge_w_parts


def _read_inter_edges(data, parent_orig_idx: torch.Tensor):
    """
    Reference inter edges are stored in the child file as (local_src, parent_dst_global) and edge_attr.
    """

    loc_t = data.inter_local_node.long()
    if loc_t.numel() == 0:
        return torch.zeros(2, 0, dtype=torch.long), None
    ext_t = data.inter_external.long()
    att_t = data.inter_edge_attr.float()
    par = parent_orig_idx.long()
    pos = torch.searchsorted(par, ext_t).clamp(max=par.numel() - 1)
    ok = par[pos] == ext_t
    if not bool(ok.all()):
        logger.warning(f"{int((~ok).sum())} external endpoints of {data.graph_id} not found in parent")

    return torch.stack([loc_t[ok], pos[ok]]), att_t[ok]


def assemble_inter_edges(
        inter_edge_model,
        inter_loader,
        device: str = 'cpu',
        noise: float = 0.0,
        source: str = 'model',
    ) -> dict:
    """
    Generate inter edges via bottom-up traversal.
    
    source: file reads from files, also version 1-4.
    versions 5-8 are full generative versions.
    """

    inter_results = {}

    # Topological order is walked once; parent_path_map built alongside.
    order = inter_loader.bottom_up_order()
    parent_path_map = {}
    for n in order:
        par = n['parent']
        if par is not None:
            # parent gid -> parent file path
            par_path = inter_loader.get_file_path(par) if hasattr(inter_loader, 'get_file_path') else None
            parent_path_map[n['graph_id']] = par_path

    parent_file_cache = {} # parent_path -> parent_orig_idx (internal nodes only)

    # file source: reference crossings
    if source == 'file':
        for node in order:
            if node['parent'] is None:
                continue # the root owns no crossings and its file holds the full x
            gid = node['graph_id']
            data = _load_subgraph(node['file_path'])
            parent_orig_idx = _parent_orig_idx(data, parent_file_cache, parent_path_map, gid)
            ei, attr = _read_inter_edges(data, parent_orig_idx)
            inter_results[gid] = {'edge_index': ei, 'edge_attr': attr}
            del data
        return inter_results

    # model source
    inter_edge_model.eval()

    with torch.no_grad():
        for node in order:
            gid = node['graph_id']
            cluster_id = node['cluster_id']

            # Each crossing is owned by a direct child of its endpoints' LCA (leaves included).
            if node['parent'] is None:
                continue

            data = _load_subgraph(node['file_path'])

            # x stays on the CPU; the model moves only its candidate rows.
            x = data.x.float()

            # Local rows in the PARENT's cluster space, as in InterEdgeDataset:
            # => leaf -> data.S_pool is already [N_local, K_parent]
            # => internal -> data.S_pool_parent is the child slice
            if hasattr(data, 'S_pool_parent') and data.S_pool_parent is not None:
                S = data.S_pool_parent.float().to(device)
            else:
                S = data.S_pool.float().to(device)
            K_global = inter_edge_model.K
            if S.shape[1] < K_global:
                S = F.pad(S, (0, K_global - S.shape[1]))

            # x_pool_parent: [K_parent, pool_dim]
            if hasattr(data, 'x_pool_parent') and data.x_pool_parent is not None:
                x_pool_parent = data.x_pool_parent.float().to(device)
            else:
                x_pool_parent = data.x_pool.float().to(device)
            if x_pool_parent.dim() == 1:
                x_pool_parent = x_pool_parent.unsqueeze(0).expand(S.shape[1], -1)

            # x_pool is the node's own pooled embedding, as in training (data.x_pool.mean(0)), not a row of the parent's table.
            x_pool = data.x_pool.float().to(device)
            if x_pool.dim() > 1:
                x_pool = x_pool.mean(dim=0)
            cid_safe = min(max(cluster_id, 0), max(S.shape[1] - 1, 0))

            # A_pool, zero-padded to [K_global, K_global] as in the training collate.
            if hasattr(data, 'A_pool_parent') and data.A_pool_parent is not None:
                A_pool = data.A_pool_parent.float().to(device)
            elif hasattr(data, 'A_pool') and data.A_pool is not None:
                A_pool = data.A_pool.float().to(device)
            else:
                A_pool = torch.zeros(S.shape[1], S.shape[1], device=device)
            if A_pool.shape[0] < K_global or A_pool.shape[1] < K_global:
                A_pool = F.pad(A_pool, (0, K_global - A_pool.shape[1], 0, K_global - A_pool.shape[0]))

            # Destination pool: all parent nodes, [N_parent, K_parent]; dst indices are parent rows, mapped to global ids in _resolve_inter_edges_to_global.
            S_pool_parent = data.S_pool_parent_full.float().to(device)
            if S_pool_parent.shape[1] < K_global:
                S_pool_parent = F.pad(S_pool_parent, (0, K_global - S_pool_parent.shape[1]))

            ei, ew = inter_edge_model.generate(
                x = x,
                S = S,
                x_pool = x_pool,
                A_pool = A_pool,
                S_pool_parent = S_pool_parent,
                x_pool_parent = x_pool_parent,
                cluster_id = cid_safe,
                noise = noise,
            )

            ei_cpu = ei.cpu() if ei is not None else torch.zeros(2, 0, dtype=torch.long)
            ew_cpu = ew.cpu() if ew is not None else None

            inter_results[gid] = {'edge_index': ei_cpu, 'edge_attr': ew_cpu}
            del data

    return inter_results


def _resolve_inter_edges_to_global(inter_results, inter_loader, leaf_meta):
    """
    Resolve (local_src, parent_dst_local) -> (global_src, global_dst). 
    """
    parent_lookup = {}
    path_lookup = {}
    for node in inter_loader.bottom_up_order():
        gid = node['graph_id']
        parent_lookup[gid] = node['parent']
        path_lookup[gid] = node['file_path']

    internal_cache = _BoundedCache(maxsize=16)

    def _orig_idx_for(gid):
        if gid in leaf_meta:
            return leaf_meta[gid]['original_idx']
        hit = internal_cache.get(gid)
        if hit is not None:
            return hit
        if gid not in path_lookup:
            return None
        data = _load_subgraph(path_lookup[gid])
        oi = data.original_node_indices.cpu()
        internal_cache.put(gid, oi)
        return oi

    src_list, dst_list, attr_list = [], [], []
    has_attr = False

    for gid, res in inter_results.items():
        ei = res['edge_index']
        attr = res.get('edge_attr', None)

        if ei.shape[1] == 0:
            continue

        orig_idx = _orig_idx_for(gid)
        if orig_idx is None:
            continue

        local_src = ei[0]

        valid = ((local_src >= 0) & (local_src < len(orig_idx)))

        if not valid.all():
            local_src = local_src[valid]
            ei = ei[:, valid]
            attr = attr[valid] if attr is not None else None

        global_src = orig_idx[local_src]

        # dst is a row of the parent's original_node_indices
        dst_orig = _orig_idx_for(parent_lookup[gid])
        parent_dst = ei[1]
        valid_dst = parent_dst < len(dst_orig)
        if not valid_dst.all():
            global_src = global_src[valid_dst]
            parent_dst = parent_dst[valid_dst]
            attr = attr[valid_dst] if attr is not None else None
        global_dst = dst_orig[parent_dst]

        src_list.append(global_src)
        dst_list.append(global_dst)
        if attr is not None:
            attr_list.append(attr)
            has_attr = True

    if not src_list:
        return (torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long), None)

    all_src = torch.cat(src_list)
    all_dst = torch.cat(dst_list)
    all_attr = torch.cat(attr_list) if has_attr else None

    return all_src, all_dst, all_attr


def _dedupe_edges(
        src: torch.Tensor,
        dst: torch.Tensor,
        attr: Optional[torch.Tensor],
        directed: bool,
        num_nodes: int,
    ):
    """
    Remove duplicate edges. Vectorized.
    """
    if src.numel() == 0:
        return src, dst, attr

    assert num_nodes < 3_000_000_000, "int64 linear edge key overflows for num_nodes >= ~3e9"

    src = src.long()
    dst = dst.long()

    if directed:
        a, b = src, dst
    else:
        a = torch.minimum(src, dst)
        b = torch.maximum(src, dst)

    key = a * num_nodes + b

    order = torch.argsort(key, stable=True)
    key_sorted = key[order]
    first_of_group = torch.ones_like(key_sorted, dtype=torch.bool)
    first_of_group[1:] = key_sorted[1:] != key_sorted[:-1]
    keep_t = order[first_of_group]
    keep_t, _ = torch.sort(keep_t) # restore input order

    if not directed:
        # Symmetrize: append reverse direction
        new_src = torch.cat([src[keep_t], dst[keep_t]])
        new_dst = torch.cat([dst[keep_t], src[keep_t]])
        if attr is not None:
            new_attr = torch.cat([attr[keep_t], attr[keep_t]])
        else:
            new_attr = None
        return (new_src, new_dst, new_attr)

    return (src[keep_t], dst[keep_t],
        attr[keep_t] if attr is not None else None)


def build_pyg_graph_streamed(x_mm_path: str, num_nodes: int, feat_dim: int, edge_src_parts: list, 
            edge_dst_parts: list, edge_w_parts: list, inter_results: dict = None, inter_loader = None,
            leaf_meta: dict = None, weighted: bool = False, directed: bool = True, unlink_backing: bool = True, ) -> Data:
    """
    Final assembly from streamed parts.
    """
    # inter-edges
    if inter_results is not None and inter_loader is not None and inter_results:
        isrc, idst, iattr = _resolve_inter_edges_to_global(inter_results, inter_loader, leaf_meta)
        if isrc.shape[0] > 0:
            edge_src_parts.append(isrc)
            edge_dst_parts.append(idst)
            if weighted and iattr is not None:
                edge_w_parts.append(iattr)

    # assemble + dedupe edges
    if edge_src_parts:
        src_all = torch.cat(edge_src_parts)
        dst_all = torch.cat(edge_dst_parts)
        edge_src_parts.clear()
        edge_dst_parts.clear()
        attr_all = torch.cat(edge_w_parts) if (weighted and edge_w_parts) else None
        edge_w_parts.clear()

        src_all, dst_all, attr_all = _dedupe_edges(src_all, dst_all, attr_all, directed=directed, num_nodes=num_nodes)
        edge_index = torch.stack([src_all, dst_all], dim=0)
        edge_weight = attr_all
    else:
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        edge_weight = None

    x_np = np.memmap(x_mm_path, dtype=np.float32, mode='r+', shape=(num_nodes, feat_dim))
    x_cat = torch.from_numpy(x_np)

    if unlink_backing:
        try:
            os.unlink(x_mm_path) # Linux: mapping stays valid
        except OSError:
            pass

    data = Data(x=x_cat, edge_index=edge_index)
    if edge_weight is not None:
        data.edge_attr = edge_weight

    # rows are scattered by global ID, so node order is 0..N-1 by construction
    data.original_node_indices = torch.arange(num_nodes, dtype=torch.long)
    data.num_nodes = num_nodes

    return data


def assemble_version(version: int, node_generator, intra_edge_model, inter_edge_model, node_dataloader, inter_loader,
        x_mean: torch.Tensor, x_std: torch.Tensor, device: str = 'cpu', node_noise: float = 0.3, edge_noise: float = 0.3,
        inter_noise: float = 0.3, weighted: bool = False, inter_source: str = 'model', workdir: Optional[str] = None, _leaf_meta_cache: dict = {}, ) -> Data:
    """
    recon -> learned reconstruction (no noise)
    gen -> learned generation (with noise)
    version  node    intra   inter
    ------------------------------
      1      recon   recon   recon
      2      gen     recon   recon
      3      recon   gen     recon
      4      gen     gen     recon
      5      recon   recon   gen
      6      gen     recon   gen
      7      recon   gen     gen
      8      gen     gen     gen
    """
    assert version in range(1, 9), "version must be 1-8"

    node_n = node_noise if version in {2, 4, 6, 8} else 0.0
    intra_n = edge_noise if version in {3, 4, 7, 8} else 0.0
    inter_n = inter_noise if version in {5, 6, 7, 8} else 0.0

    directed = _infer_directed(intra_edge_model, default=True)

    ds_key = id(node_dataloader.dataset)
    if ds_key not in _leaf_meta_cache:
        _leaf_meta_cache.clear() 
        _leaf_meta_cache[ds_key] = scan_leaf_meta(node_dataloader.dataset)
    leaf_meta, num_nodes = _leaf_meta_cache[ds_key]

    feat_dim = int(x_mean.shape[0])

    if workdir is None:
        workdir = tempfile.gettempdir()
    os.makedirs(workdir, exist_ok=True)
    fd, x_mm_path = tempfile.mkstemp(suffix='.f32', prefix='assemble_x_', dir=workdir)
    os.close(fd)
    x_mm = np.memmap(x_mm_path, dtype=np.float32, mode='w+', shape=(num_nodes, feat_dim))

    edge_src_parts, edge_dst_parts, edge_w_parts = assemble_leaves_streaming(
            node_generator = node_generator,
            intra_edge_model = intra_edge_model,
            dataloader = node_dataloader,
            x_mean = x_mean,
            x_std = x_std,
            leaf_meta = leaf_meta,
            x_mm = x_mm,
            device = device,
            node_noise = node_n,
            intra_noise = intra_n,
            weighted = weighted,
        )
    del x_mm # flushed inside; reopened in build_pyg_graph_streamed

    if inter_edge_model is None:
        inter_results = {} # only intra
    else:
        inter_results = assemble_inter_edges( inter_edge_model = inter_edge_model, inter_loader = inter_loader, device = device, noise = inter_n, source = inter_source if version in {1, 2, 3, 4} else 'model', )

    return build_pyg_graph_streamed(
        x_mm_path = x_mm_path,
        num_nodes = num_nodes,
        feat_dim = feat_dim,
        edge_src_parts = edge_src_parts,
        edge_dst_parts = edge_dst_parts,
        edge_w_parts = edge_w_parts,
        inter_results = inter_results,
        inter_loader = inter_loader,
        leaf_meta = leaf_meta,
        weighted = weighted,
        directed = directed,
    )