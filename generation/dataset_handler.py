import json
import os
import random
import logging
from collections import defaultdict, deque

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


from inter_edge_generator import soft_entropy, topk_bridge_candidates

logger = logging.getLogger(__name__)


def load_leaf_meta(json_path: str):
    """
    Return (paths, meta, raw) for the leaf nodes of the cluster mapping JSON.
    """
    with open(json_path, 'r') as f:
        raw = json.load(f)

    paths, meta = [], []
    for gid, info in raw['graphs'].items():
        if not info['is_leaf']:
            continue
        meta.append({'graph_id': gid, 'level': info['level'], 'parent': info['parent'], })
        paths.append(info['file_path'])

    logger.info(f"Found {len(meta)} leaf graphs")
    return paths, meta, raw


def load_all_meta(json_path: str):
    """
    Walk the cluster mapping JSON and return (paths, meta) for ALL nodes (leaves + internal + root). 
    Used by the inter-edge loader.
    """
    with open(json_path, 'r') as f:
        raw = json.load(f)

    paths, meta = [], []
    gid_to_meta = {}
    for gid, info in raw['graphs'].items():
        m = {'graph_id': gid,
            'level': info.get('level', 0),
            'parent': info.get('parent', None),
            'cluster_id': info.get('cluster_id', -1),
            'file_path': info['file_path'],
            'is_leaf': info.get('is_leaf', False), }
        meta.append(m)
        paths.append(info['file_path'])
        gid_to_meta[gid] = m

    logger.info(f"Found {len(meta)} subgraphs (all levels)")
    return paths, meta, raw, gid_to_meta


def resolve_global_max_K(json_path: str, raw: dict, paths: list) -> int:
    """
    Read '_global_max_K' from the JSON cache or compute and save it by scanning all .pt files. 
    """
    if '_global_max_K' in raw:
        K = raw['_global_max_K']
        logger.info(f"Loaded global_max_K from cache: {K}")
        return K

    logger.info("Computing global_max_K (one-time scan)...")
    K = max(torch.load(p, weights_only=False).S_pool.shape[1] for p in paths)
    raw['_global_max_K'] = K

    with open(json_path, 'w') as f:
        json.dump(raw, f)

    logger.info(f"Global max_K computed, saved: {K}")

    return K


def resolve_feature_stats(json_path: str, paths: list):
    """
    Return (x_mean, x_std).
    If it is not in the files, compute it by scanning over the .pt files and cache it.
    """
 
    stats_path = json_path.replace('.json', '_feature_stats.pt')
 
    if os.path.exists(stats_path):
        logger.info(f"Loading cached feature stats from {stats_path}")
        stats = torch.load(stats_path, weights_only=True)
        return stats['mean'], stats['std']
 
    logger.info("Computing feature stats (one-time scan)...")
    n = 0
    s = None
    s_sq = None
    for p in paths:
        x = torch.load(p, weights_only=False).x.double()
        if s is None:
            s = torch.zeros(x.shape[1], dtype=torch.float64)
            s_sq = torch.zeros(x.shape[1], dtype=torch.float64)
        n += x.shape[0]
        s += x.sum(0)
        s_sq += (x * x).sum(0)
        del x
 
    x_mean = (s / n)
    var = (s_sq - n * x_mean * x_mean) / (n - 1)
    x_std = var.clamp(min=0).sqrt().clamp(min=1e-6).float()
    x_mean = x_mean.float()
 
    torch.save({'mean': x_mean, 'std': x_std}, stats_path)
    logger.info(f"Feature stats cached to {stats_path}")
 
    return x_mean, x_std


def resolve_feature_type(json_path: str, paths: list, override: str = 'auto') -> str:
    """
    Return 'binary' or 'continuous' for the reference graph's node features.
    """

    if override in ('binary', 'continuous'):
        logger.info(f"Feature type forced to '{override}'")
        return override

    cache_path = json_path.replace('.json', '_feature_type.txt')
    if os.path.exists(cache_path):
        with open(cache_path, 'r') as f:
            ft = f.read().strip()
        if ft in ('binary', 'continuous'):
            logger.info(f"Loaded feature type from cache: {ft}")
            return ft

    ft = 'binary'
    # if any feature vector has a non-binary value, treat the whole dataset as continuous.
    # if hybrid, still treated as continuous.
    for p in paths:
        x = torch.load(p, weights_only=False).x
        if x.numel() and not bool(((x == 0) | (x == 1)).all()):
            ft = 'continuous'
            del x
            break
        del x

    with open(cache_path, 'w') as f:
        f.write(ft)

    logger.info(f"Detected feature type: {ft}")
    return ft


def resolve_inter_edge_owners(json_path: str, paths: list, meta: list):
    """
    Return (train_paths, train_meta): the subset of subgraphs that own at least one inter-cluster edge.

    The scan opens every .pt once, so the resulting graph_id list is cached in file next to the mapping JSON and reused on later runs. 
    
    This is especially necessary for large datasets.
    """
    cache_path = json_path.replace('.json', '_inter_edge_owners.json')

    owners = None
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r') as f:
                cached = json.load(f)
            if cached.get('num_subgraphs') == len(paths):
                owners = set(cached['owners'])
                logger.info(f"Loaded inter-edge owner list from cache: {len(owners)} subgraphs")
            else:
                logger.warning(f"Inter-edge owner cache is stale ({cached.get('num_subgraphs')} vs {len(paths)} subgraphs), rescanning")
        except (json.JSONDecodeError, KeyError, OSError) as err:
            logger.warning(f"Could not read inter-edge owner cache ({err}); rescanning")

    if owners is None:
        logger.info(f"Scanning {len(paths)} subgraphs for inter-cluster edges (one-time)...")
        owners = set()
        for p, m in zip(paths, meta):
            d = torch.load(p, weights_only=False)
            if d.inter_local_node.numel() > 0:
                owners.add(m['graph_id'])
            del d
        try:
            with open(cache_path, 'w') as f:
                json.dump({'num_subgraphs': len(paths), 'owners': sorted(owners)}, f)
            logger.info(f"Inter-edge owner list cached to {cache_path}")
        except OSError as err:
            logger.warning(f"Could not write inter-edge owner cache ({err})")

    train_paths, train_meta = [], []
    for p, m in zip(paths, meta):
        if m['graph_id'] in owners:
            train_paths.append(p)
            train_meta.append(m)

    logger.info(f"Inter training set: {len(train_paths)}/{len(paths)} subgraphs carry inter edges")
    return train_paths, train_meta


def pad_pairs(t: torch.Tensor, target_len: int) -> torch.Tensor:
    """Pad [E, 2] pair tensor to [target_len, 2]."""
    current = t.shape[0]
    assert current <= target_len, f"Current length {current} exceeds target {target_len}"

    if current == target_len:
        return t
    if current == 0:
        return torch.zeros(target_len, 2, dtype=torch.long)
    
    return F.pad(t, (0, 0, 0, target_len - current))


def pad_weights(t: torch.Tensor, target_len: int) -> torch.Tensor:
    """Pad [E] weight tensor to [target_len]."""
    current = t.shape[0]
    assert current <= target_len, f"Current length {current} exceeds target {target_len}"

    if current == target_len:
        return t
    if current == 0:
        return torch.zeros(target_len, dtype=torch.float)
    
    return F.pad(t, (0, target_len - current))


def topological_bottom_up(mapping: dict) -> list:
    """Return graph_ids in bottom-up topological order (leaves first, root last)."""
    
    children = defaultdict(list)
    for gid, info in mapping.items():
        parent = info.get('parent', None)
        if parent:
            children[parent].append(gid)

    num_children = {gid: len(children[gid]) for gid in mapping}

    queue = deque([gid for gid, deg in num_children.items() if deg == 0])
    order = []

    while queue:
        gid = queue.popleft()
        order.append(gid)
        parent = mapping[gid].get('parent', None)
        if parent and parent in num_children:
            num_children[parent] -= 1
            if num_children[parent] == 0:
                queue.append(parent)

    if len(order) != len(mapping):
        logger.warning(f"Topological sort incomplete: {len(order)}/{len(mapping)}. Please rerun clustering.")

    return order


def build_children_map(mapping: dict) -> dict:
    """Build parent_gid -> list[child_gid]."""

    children_map = defaultdict(list)
    for gid, info in mapping.items():
        parent = info.get('parent', None)
        if parent:
            children_map[parent].append(gid)

    return children_map


class BucketedBatchSampler(torch.utils.data.Sampler):
    """
    Group similar-sized leaves into the same batch.

    This batching enhances speed a lot, so we wont pad a lot.

    Idea: check by .pt sizes, as the files have similar metadata. So similar in size, similar in number of nodes.
    """

    def __init__(self, file_paths: list, batch_size: int, shuffle: bool = True):
        sizes = [os.path.getsize(p) for p in file_paths]
        order = sorted(range(len(sizes)), key=lambda i: sizes[i])
        self.batches = [order[i:i + batch_size] for i in range(0, len(order), batch_size)]
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            for i in torch.randperm(len(self.batches)).tolist():
                yield self.batches[i]
        else:
            for b in self.batches:
                yield b

    def __len__(self):
        return len(self.batches)



class LeafGraphDataset(Dataset):
    """Each sample is a leaf graph with its x, S_pool, and x_pool. Used by the node generator."""
    def __init__(self, file_paths: list, meta: list):
        self.file_paths = file_paths
        self.meta = meta

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        graph = torch.load(self.file_paths[idx], weights_only=False)
        return {
            'x_pool': graph.x_pool,
            'S_pool': graph.S_pool,
            'x': graph.x,
            'graph_id': self.meta[idx]['graph_id'],
            'level': self.meta[idx]['level'],
            'parent': self.meta[idx]['parent'],
        }


def make_node_collate_fn(global_max_K: int):
    """
    Collate leaf samples for the node generator.

    All these collate functions are required, as we are training with variable-sized communities.    
    """

    def collate(batch):
        max_N = max(b['x'].shape[0] for b in batch)

        x_pool_list, S_list, x_list, mask_list = [], [], [], []

        for b in batch:
            N, K = b['S_pool'].shape
            # F.pad last-dim-first: (K_left, K_right, N_left, N_right)
            S_padded = F.pad(b['S_pool'], (0, global_max_K - K, 0, max_N - N))
            x_padded = F.pad(b['x'], (0, 0, 0, max_N - N))
            mask = torch.arange(max_N) < N

            x_pool_list.append(b['x_pool'])
            S_list.append(S_padded)
            x_list.append(x_padded)
            mask_list.append(mask)

        return {
            'x_pool': torch.stack(x_pool_list),
            'S': torch.stack(S_list),
            'x': torch.stack(x_list),
            'mask': torch.stack(mask_list),
            }

    return collate


class NodeDataLoader:
    def __init__(self, args):
        self.args = args
        self.json_path = args.json_path
        self.leaf_meta = []
        self.leaf_paths = []
        self.dataloader = None
        self.global_max_K = None
        self.x_mean = None
        self.x_std = None

    def load_data(self):
        self.leaf_paths, self.leaf_meta, raw = load_leaf_meta(self.json_path)
        self.global_max_K = resolve_global_max_K(self.json_path, raw, self.leaf_paths)
        self.x_mean, self.x_std = resolve_feature_stats(self.json_path, self.leaf_paths)

        self.feature_type = resolve_feature_type(self.json_path, self.leaf_paths, getattr(self.args, 'feature_type', 'auto'))
        self.args.feature_type = self.feature_type

        order = sorted(range(len(self.leaf_paths)), key=lambda i: os.path.getsize(self.leaf_paths[i]))
        self.leaf_paths = [self.leaf_paths[i] for i in order]
        self.leaf_meta = [self.leaf_meta[i] for i in order]

        dataset = LeafGraphDataset(self.leaf_paths, self.leaf_meta)
        
        self.dataloader = DataLoader(
            dataset,
            batch_size = self.args.batch_size,
            shuffle = False,
            collate_fn = make_node_collate_fn(self.global_max_K),
            )

        return self.dataloader


class IntraEdgeDataset(Dataset):
    def __init__(
            self,
            file_paths: list,
            meta: list,
            neg_ratio: float = 1.0,
            directed: bool = False,
            weighted: bool = False,
            max_pairs = 10000,        
            neg_pool_frac: float = 0.01,
            max_neg: int = 20000,
        ):

        self.file_paths = file_paths
        self.meta = meta
        self.neg_ratio = neg_ratio
        self.directed = directed
        self.weighted = weighted
        self.max_pairs = int(max_pairs)
        self.neg_pool_frac = float(neg_pool_frac)
        self.max_neg = int(max_neg)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        leaf = torch.load(self.file_paths[idx], weights_only=False)

        N = leaf.num_nodes
        edge_index_full = leaf.edge_index
        x = leaf.x
        S_pool = leaf.S_pool
        x_pool = leaf.x_pool

        # degree calculation, doing like below, so if we deal with directed graphs, it should also be fine.
        src, dst = edge_index_full[0], edge_index_full[1]
        in_degree = torch.zeros(N, dtype=torch.float)
        out_degree = torch.zeros(N, dtype=torch.float)
        in_degree.scatter_add_(0, dst, torch.ones_like(dst, dtype=torch.float))
        out_degree.scatter_add_(0, src, torch.ones_like(src, dtype=torch.float))
        degree = torch.stack([in_degree, out_degree], dim=1)

        # edge weights: use edge_attr if available and weighted=True, otherwise default to 1.0
        # did not tested code on weighted graphs, but it should work?
        edge_index = edge_index_full
        E_full = edge_index_full.shape[1]

        if (self.weighted and hasattr(leaf, 'edge_attr') and leaf.edge_attr is not None and leaf.edge_attr.numel() > 0):
            ea = leaf.edge_attr
            if ea.dim() == 1:
                edge_weight = ea.float()
            elif ea.shape[1] == 1:
                edge_weight = ea.squeeze(1).float()
            else:
                # Multi-dimensional edge_attr is reduced to its L2 norm.
                edge_weight = ea.norm(dim=1).float()
        else:
            edge_weight = torch.ones(E_full, dtype=torch.float)

        # Undirected: keep u < v only. Assumes edge_index stores both directions; drops self-loops and the (v, u) copy.
        if not self.directed and E_full > 0:
            mask = edge_index[0] < edge_index[1] # upper triangle filter for undirected graphs
            edge_index = edge_index[:, mask]
            edge_weight = edge_weight[mask]

        E = edge_index.shape[1]

        if E == 0:
            pos_pairs = torch.zeros(0, 2, dtype=torch.long)
            neg_pairs = torch.zeros(0, 2, dtype=torch.long)
            edge_weight = torch.zeros(0, dtype=torch.float)
        else:
            pos_pairs = edge_index.T.long()

            # max_pairs is required for GPU memory control. It is all fine on small graphs. 
            # But some communities will make intra edge training or generation OOM.            
            if self.max_pairs > 0 and E > self.max_pairs:
                keep = torch.randperm(E)[:self.max_pairs]
                pos_pairs = pos_pairs[keep]
                edge_weight = edge_weight[keep]
                E = self.max_pairs

            num_neg = int(E * self.neg_ratio)
            neg_pairs = self._sample_negatives(N, pos_pairs, num_neg)

        # edge_index is always [2, E] for the positive edges only. The negative edges are in neg_pairs.
        edge_index = pos_pairs.T.contiguous() if pos_pairs.shape[0] > 0 else torch.zeros(2, 0, dtype=torch.long)

        return {
            'x': x,
            'S_pool': S_pool,
            'x_pool': x_pool,
            'pos_pairs': pos_pairs,
            'neg_pairs': neg_pairs,
            'edge_index': edge_index,
            'edge_weight': edge_weight,
            'degree': degree,
            'num_nodes': N,
            'graph_id': self.meta[idx]['graph_id'],
            'level': self.meta[idx]['level'],
            'parent': self.meta[idx]['parent'],
            }

    def _sample_negatives(self, N, pos_pairs, num_neg):
        """
        Standard negative sampling: sample random (u, v) pairs that are not in the positive set.

        Also some operations for directed graphs as well.
        """
        num_neg = int(num_neg)
        pos_set = set(map(tuple, pos_pairs.tolist()))
        if not self.directed:
            pos_set |= {(v, u) for (u, v) in pos_set}

        negs = []
        attempts = 0
        max_attempts = num_neg * 10

        while len(negs) < num_neg and attempts < max_attempts:
            src = torch.randint(0, N, (num_neg,))
            dst = torch.randint(0, N, (num_neg,))
            for s, d in zip(src.tolist(), dst.tolist()):
                if s != d and (s, d) not in pos_set:
                    negs.append([s, d])
                    pos_set.add((s, d))   # avoid duplicate negatives
                    if len(negs) == num_neg:
                        break
            attempts += num_neg

        if len(negs) < num_neg:
            # This happens for some graphs that has many components. We do not correct it. Lets say you have a graph with 3 nodes and all connected, we cannot sample negative.
            logger.warning(f"IntraEdge negative sampling undersampled: {len(negs)}/{num_neg} (N={N}, E={len(pos_pairs)})")

        return torch.tensor(negs, dtype=torch.long) if negs else torch.zeros(0, 2, dtype=torch.long)


def make_intra_edge_collate_fn(global_max_K: int, degree_mode: str = 'directed'):
    """
    Collate intra-edge samples for the intra-edge generator.
    """
    assert degree_mode in ('directed', 'undirected'), f"degree_mode must be 'directed' or 'undirected', got {degree_mode}"

    def collate(batch):
        max_N = max(b['num_nodes'] for b in batch)
        max_E = max(b['pos_pairs'].shape[0] for b in batch)
        max_neg = max(b['neg_pairs'].shape[0] for b in batch)

        # Guard against fully-empty batches so shapes are always valid. 
        # I think this can happen if a batch has only empty graphs and neg_ratio=0.
        max_E = max(max_E, 1)
        max_neg = max(max_neg, 1)

        x_list, S_list, xpool_list = [], [], []
        pos_list, neg_list, ew_list = [], [], []
        deg_list, nmask_list = [], []
        emask_list, nmask_neg_list = [], []

        for b in batch:
            N = b['num_nodes']
            K = b['S_pool'].shape[1]
            E = b['pos_pairs'].shape[0]
            Eg = b['neg_pairs'].shape[0]

            # Just in case, some sanity checks. 
            assert K <= global_max_K, f"S_pool K ({K}) exceeds global_max_K ({global_max_K})"
            assert E <= max_E, f"Number of positive edges ({E}) exceeds max_E ({max_E}) in the batch"
            assert Eg <= max_neg, f"Number of negative edges ({Eg}) exceeds max_neg ({max_neg}) in the batch"

            x_pad = F.pad(b['x'], (0, 0, 0, max_N - N))
            S_pad = F.pad(b['S_pool'], (0, global_max_K - K, 0, max_N - N))
            deg_pad = F.pad(b['degree'], (0, 0, 0, max_N - N))
            nmask = torch.arange(max_N) < N

            pos_pad = pad_pairs(  b['pos_pairs'], max_E)
            neg_pad = pad_pairs(b['neg_pairs'], max_neg)
            ew_pad = pad_weights(b['edge_weight'], max_E)

            emask = torch.arange(max_E) < E # real positives
            nmask_n = torch.arange(max_neg) < Eg # real negatives  

            x_list.append(x_pad)
            S_list.append(S_pad)
            xpool_list.append(b['x_pool'])
            pos_list.append(pos_pad)
            neg_list.append(neg_pad)
            ew_list.append(ew_pad)
            deg_list.append(deg_pad)
            nmask_list.append(nmask)
            emask_list.append(emask)
            nmask_neg_list.append(nmask_n)

        return {
            'x': torch.stack(x_list),
            'S': torch.stack(S_list),
            'x_pool': torch.stack(xpool_list),
            'pos_pairs': torch.stack(pos_list),
            'neg_pairs': torch.stack(neg_list),
            'edge_weight': torch.stack(ew_list),
            'degree': torch.stack(deg_list),
            'node_mask': torch.stack(nmask_list),
            'edge_mask': torch.stack(emask_list), # positives mask
            'neg_mask': torch.stack(nmask_neg_list), # negatives mask
            'edge_index_list': [b['edge_index'] for b in batch],
            'degree_mode': degree_mode, # 'directed' | 'undirected' # might change it to boolean later
            }

    return collate


class IntraEdgeDataLoader:
    def __init__(self, args):
        self.args = args
        self.json_path = args.json_path
        self.directed = getattr(args, 'directed', True)
        self.weighted = getattr(args, 'weighted', True)
        self.neg_ratio = getattr(args, 'neg_ratio', 1)

        self.max_pairs = getattr(args, 'intra_max_pairs', 50000)

        self.leaf_meta = []
        self.leaf_paths = []
        self.dataloader = None
        self.global_max_K = None
        self.x_mean = None
        self.x_std = None

    def load_data(self):
        self.leaf_paths, self.leaf_meta, raw = load_leaf_meta(self.json_path)
        self.global_max_K = resolve_global_max_K(self.json_path, raw, self.leaf_paths)
        self.x_mean, self.x_std = resolve_feature_stats(self.json_path, self.leaf_paths)

        dataset = IntraEdgeDataset(
            file_paths = self.leaf_paths,
            meta = self.leaf_meta,
            neg_ratio = self.neg_ratio,
            directed = self.directed,
            weighted = self.weighted,
            max_pairs = self.max_pairs
        )

        self.dataloader = DataLoader(
            dataset,
            batch_sampler = BucketedBatchSampler(self.leaf_paths, self.args.batch_size, shuffle=True),
            collate_fn  = make_intra_edge_collate_fn(self.global_max_K, degree_mode = 'directed' if self.directed else 'undirected',),
            num_workers = getattr(self.args, 'num_workers', 0),
        )

        return self.dataloader

class InterEdgeDataset(Dataset):
    """
    Each sample is a subgraph with its x, S_pool, x_pool, and inter-cluster edges.
    Used by the inter-edge generator.
    """
    def __init__(
            self,
            file_paths: list,
            meta: list,
            neg_ratio: float = 1.0,
            s_threshold: float = 0.1,
            topk_nodes: int = 50,
            directed: bool = False,
            max_pairs: int = 10000,
            neg_pool_frac: float = 0.01,
            max_neg: int = 20000,
        ):

        self.file_paths = file_paths
        self.meta = meta
        self.neg_ratio = neg_ratio
        self.s_threshold = s_threshold
        self.topk_nodes = topk_nodes
        self.directed = directed
        self.max_pairs = int(max_pairs)

        self.neg_pool_frac = float(neg_pool_frac)
        self.max_neg = int(max_neg)


    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        data = torch.load(path, weights_only=False)

        N = data.num_nodes
        cluster_id = data.cluster_id
        graph_id = data.graph_id
        level = data.level
        is_leaf = data.is_leaf

        x = data.x.float()
        x_pool = data.x_pool.float()
        if x_pool.dim() > 1:
            x_pool = x_pool.mean(dim=0)  # [emb_dim]

        # LOCAL src membership: rows = this subgraph's own nodes, cols = parent K.
        #   leaf -> data.S_pool is already in pt [N_local, K_parent]
        #   internal -> data.S_pool_parent is the child-slice [N_local, K_parent]
        #   root -> own S, cluster_id == -1 (unused)
        if hasattr(data, 'S_pool_parent') and data.S_pool_parent is not None:
            S = data.S_pool_parent.float() # internal: [N_local, K_parent]
        else:
            S = data.S_pool.float() # leaf: parent view; root: own. It doesnt really matter for root, but we keep it so it doesnt gives an error in code for root..

        K_self = S.shape[1] # == K_parent

        if hasattr(data, 'x_pool_parent') and data.x_pool_parent is not None:
            x_pool_parent = data.x_pool_parent.float()
        else:
            x_pool_parent = x_pool.unsqueeze(0).expand(K_self, -1)

        # A_pool must match S's (parent) cluster space. Some sanity checks.
        if hasattr(data, 'A_pool_parent') and data.A_pool_parent is not None:
            A_pool = data.A_pool_parent.float()  # [K_parent, K_parent]
        elif (hasattr(data, 'A_pool') and data.A_pool is not None
              and data.A_pool.dim() == 2 and data.A_pool.shape[0] == K_self):
            A_pool = data.A_pool.float() # root: own A_pool
        else:
            A_pool = torch.zeros(K_self, K_self)

        assert cluster_id < S.shape[1]
        assert cluster_id < A_pool.shape[0]

        # Membership of ALL parent nodes in the parent's K clusters 
        if hasattr(data, 'S_pool_parent_full') and data.S_pool_parent_full is not None:
            S_pool_parent = data.S_pool_parent_full.float() # [N_parent, K_parent]
        else:
            S_pool_parent = S.clone() # leaf/root fallback


        # Crossing endpoints: inter_local_node is already a row of this subgraph; inter_external is a global id, converted to a row of S_pool_parent
        pos_pairs, pos_attrs = [], []
        loc_t = data.inter_local_node.long()
        if loc_t.numel() > 0:
            par_orig = getattr(data, 'parent_original_node_indices', None)
            if par_orig is None:
                raise ValueError(f"graph {graph_id} has inter edges but no parent_original_node_indices")
            par_orig = par_orig.long()
            ext_t = data.inter_external.long()
            att_t = data.inter_edge_attr.float()
            pos = torch.searchsorted(par_orig, ext_t).clamp(max=par_orig.numel() - 1)
            ok = par_orig[pos] == ext_t
            if not bool(ok.all()):
                logger.warning(f"{int((~ok).sum())} external nodes in graph {graph_id} not mappable to parent-local.")
            pos_pairs = list(zip(loc_t[ok].tolist(), pos[ok].tolist()))
            pos_attrs = att_t[ok].tolist()


        # Drop exact repeats of (local row, parent row).
        if len(pos_pairs) > 0:
            merged = {}
            for (u, v), a in zip(pos_pairs, pos_attrs):
                merged[(u, v)] = a # keeps last on exact repeat
            pos_pairs = list(merged.keys())
            pos_attrs = list(merged.values())


        # max_pair budget, same thing with intra.
        if self.max_pairs > 0 and len(pos_pairs) > self.max_pairs:
            keep = torch.randperm(len(pos_pairs))[:self.max_pairs].tolist()
            pos_pairs = [pos_pairs[i] for i in keep]
            pos_attrs = [pos_attrs[i] for i in keep]

        # negative sampling
        K = S.shape[1]
        pos_set = set(map(tuple, pos_pairs))

        # Local candidates, bridge nodes
        H_loc = soft_entropy(S)
        if 0 <= cluster_id < K:
            local_cands = topk_bridge_candidates(S, cluster_id, self.s_threshold, self.topk_nodes, H=H_loc).tolist()
        else:
            # root fallback (cluster_id == -1): no parent cluster to filter on
            cand = (S.max(dim=1).values > self.s_threshold).nonzero(as_tuple=True)[0]
            if cand.numel() > self.topk_nodes:
                cand = cand[H_loc[cand].topk(self.topk_nodes).indices]
            local_cands = cand.tolist()

        # external candidates: sibling-cluster nodes from S_pool_parent
        connected = []
        if 0 <= cluster_id < A_pool.shape[0]:
            connected = (A_pool[cluster_id] > 0).nonzero(as_tuple=True)[0].tolist()

        all_ext_cands = []
        H_par = soft_entropy(S_pool_parent)
        for c_other in connected:
            if c_other == cluster_id or c_other >= S_pool_parent.shape[1]:
                continue
            above = topk_bridge_candidates(S_pool_parent, c_other, self.s_threshold, self.topk_nodes, H=H_par, exclude_cluster=cluster_id)
            all_ext_cands.extend(above.tolist())
        all_ext_cands = list(set(all_ext_cands))

        # Negatives scale with the candidate pool that generation scores, not only # with the number of positives
        pool_size = len(local_cands) * len(all_ext_cands)
        n_neg = max(int(len(pos_pairs) * self.neg_ratio), int(self.neg_pool_frac * pool_size))
        n_neg = min(n_neg, self.max_neg)
        neg_pairs = []
        attempts = 0
        max_att = n_neg * 20
        seen = set(pos_set)

        while len(neg_pairs) < n_neg and attempts < max_att:
            if attempts == (max_att - 1):
                logger.warning(f"InterEdge negative sampling undersampled: {len(neg_pairs)}/{n_neg} after {attempts} attempts (local_cands={len(local_cands)}, "
                            f" all_ext_cands={len(all_ext_cands)}, pos_pairs={len(pos_pairs)}, cluster_id={cluster_id}, graph_id={graph_id})" )
                
            if not local_cands or not all_ext_cands:
                break
            i = random.choice(local_cands)
            j = random.choice(all_ext_cands)
            if (i, j) not in seen:
                neg_pairs.append((i, j))
                seen.add((i, j))
            attempts += 1

        pos_t = torch.tensor(pos_pairs, dtype=torch.long) if pos_pairs else torch.zeros(0, 2, dtype=torch.long)
        pos_a = torch.tensor(pos_attrs, dtype=torch.float) if pos_attrs else torch.zeros(0, dtype=torch.float)
        neg_t = torch.tensor(neg_pairs, dtype=torch.long) if neg_pairs else torch.zeros(0, 2, dtype=torch.long)

        if hasattr(data, 'edge_index') and data.edge_index is not None:
            edge_index = data.edge_index.long()
        else:
            edge_index = torch.zeros(2, 0, dtype=torch.long)

        return {
            'x': x,
            'S': S,
            'x_pool': x_pool,
            'A_pool': A_pool,
            'S_pool_parent': S_pool_parent,
            'x_pool_parent': x_pool_parent,
            'pos_pairs': pos_t,
            'neg_pairs': neg_t,
            'pos_attr': pos_a,
            'edge_index': edge_index,
            'cluster_id': cluster_id,
            'graph_id': graph_id,
            'level': level,
            'is_leaf': int(is_leaf),
            'N': N, 
            }


def make_inter_edge_collate_fn(global_K: int):
    """ Create a collate function for inter-edge sampling"""

    def collate(samples):
        B = len(samples)
        max_N = max(s['N'] for s in samples)
        max_N_parent = max(s['S_pool_parent'].shape[0] for s in samples)
        max_E_pos = max(max(s['pos_pairs'].shape[0] for s in samples), 1)
        max_E_neg = max(max(s['neg_pairs'].shape[0] for s in samples), 1)
        feat_dim = samples[0]['x'].shape[1]
        emb_dim = samples[0]['x_pool_parent'].shape[1]

        x_b = torch.zeros(B, max_N, feat_dim)
        S_b = torch.zeros(B, max_N, global_K)
        xp_b = torch.zeros(B, emb_dim)
        Ap_b = torch.zeros(B, global_K, global_K)
        Sp_par_b = torch.zeros(B, max_N_parent, global_K)
        xp_par_b = torch.zeros(B, global_K, emb_dim)
        pos_b = torch.zeros(B, max_E_pos, 2, dtype=torch.long)
        neg_b = torch.zeros(B, max_E_neg, 2, dtype=torch.long)
        pos_attr_b = torch.zeros(B, max_E_pos)
        node_mask = torch.zeros(B, max_N, dtype=torch.bool)
        pos_mask = torch.zeros(B, max_E_pos, dtype=torch.bool)
        neg_mask = torch.zeros(B, max_E_neg, dtype=torch.bool)

        cluster_ids, graph_ids, levels, is_leaves = [], [], [], []

        for i, s in enumerate(samples):
            N = s['N']
            K_i = s['S'].shape[1]
            N_par = s['S_pool_parent'].shape[0]
            E_pos = s['pos_pairs'].shape[0]
            E_neg = s['neg_pairs'].shape[0]

            assert K_i <= global_K, f"S K ({K_i}) > global_K ({global_K})"
            assert s['A_pool'].shape == (K_i, K_i), f"A_pool shape {s['A_pool'].shape} does not match expected ({K_i}, {K_i})"
            assert s['x_pool_parent'].shape[0] <= global_K, f"x_pool_parent K ({s['x_pool_parent'].shape[0]}) > global_K ({global_K})"
            
            K_par = s['x_pool_parent'].shape[0]

            x_b[i, :N] = s['x']
            S_b[i, :N, :K_i] = s['S']
            Ap_b[i, :K_i, :K_i] = s['A_pool']
            Sp_par_b[i, :N_par, :K_par] = s['S_pool_parent'][:, :K_par]
            xp_par_b[i, :K_par] = s['x_pool_parent']
            xp_b[i] = s['x_pool']

            if E_pos > 0:
                pos_b[i, :E_pos] = s['pos_pairs']
                pos_attr_b[i, :E_pos] = s['pos_attr']
                pos_mask[i, :E_pos] = True
            if E_neg > 0:
                neg_b[i, :E_neg] = s['neg_pairs']
                neg_mask[i, :E_neg] = True

            node_mask[i, :N] = True
            cluster_ids.append(s['cluster_id'])
            graph_ids.append(s['graph_id'])
            levels.append(s['level'])
            is_leaves.append(s['is_leaf'])

        return {
            'x': x_b,
            'S': S_b,
            'x_pool': xp_b,
            'A_pool': Ap_b,
            'S_pool_parent': Sp_par_b,
            'x_pool_parent': xp_par_b,
            'pos_pairs': pos_b,
            'neg_pairs': neg_b,
            'pos_attr': pos_attr_b,
            'node_mask': node_mask,
            'pos_mask': pos_mask,
            'neg_mask': neg_mask,
            'cluster_id': torch.tensor(cluster_ids, dtype=torch.long),
            'graph_id': graph_ids,
            'level': torch.tensor(levels, dtype=torch.long),
            'is_leaf': torch.tensor(is_leaves, dtype=torch.bool),
        }

    return collate


class InterEdgeDataLoader:
    """
    Train : load_data() -> flat DataLoader (subgraphs owning inter edges, shuffled)
    Generation: bottom_up_order() -> graph_ids in topological order (leaves first, root last)
    """

    def __init__(self, args):
        self.args = args
        self.json_path = args.json_path
        self.neg_ratio = getattr(args, 'neg_ratio', 1.0)
        self.s_threshold = getattr(args, 's_threshold', 0.1)
        self.topk_nodes = getattr(args, 'topk_nodes', 50)
        self.directed = getattr(args, 'directed', False)
        self.max_pairs = getattr(args, 'inter_max_pairs', 10000)
        self.neg_pool_frac = getattr(args, 'inter_neg_pool_frac', 0.01)
        self.max_neg = getattr(args, 'inter_max_neg', 20000)

        self.all_meta = []
        self.all_paths = []
        self.train_meta = []
        self.train_paths = []
        self.mapping = {}
        self.global_max_K = None
        self.dataloader = None
        self.dataset = None

        self._topo_order  = None
        self._gid_to_meta = {}
        self._children_map = {}

    def load_data(self):
        self.all_paths, self.all_meta, raw, self._gid_to_meta = load_all_meta(self.json_path)
        self.mapping = raw['graphs']

        self._topo_order = topological_bottom_up(self.mapping)
        self._children_map = build_children_map(self.mapping)
        logger.info(f"Topological order: {len(self._topo_order)} nodes, first={self._topo_order[0]}, last={self._topo_order[-1]}")

        self.global_max_K = resolve_global_max_K(self.json_path, raw, self.all_paths)

        # Train only on subgraphs that own inter-cluster edges, others do not contribute anything.
        self.train_paths, self.train_meta = resolve_inter_edge_owners(self.json_path, self.all_paths, self.all_meta)
        if not self.train_paths:
            raise RuntimeError(f"No subgraph in {self.json_path} carries inter-cluster edges, rerun clustering.")

        self.dataset = InterEdgeDataset(
                file_paths = self.train_paths,
                meta = self.train_meta,
                neg_ratio = self.neg_ratio,
                s_threshold = self.s_threshold,
                topk_nodes = self.topk_nodes,
                directed = self.directed,
                max_pairs = self.max_pairs,
                neg_pool_frac = self.neg_pool_frac,
                max_neg = self.max_neg,
            )

        self.dataloader = DataLoader(
                self.dataset,
                batch_size = self.args.batch_size,
                shuffle = True,
                collate_fn = make_inter_edge_collate_fn(self.global_max_K),
                num_workers = getattr(self.args, 'num_workers', 0),
            )

        return self.dataloader

    def bottom_up_order(self) -> list:
        """
        List of dicts in bottom-up topological order (leaves first).
        Each dict: {graph_id, file_path, level, parent, cluster_id, is_leaf, children}.
        """
        if self._topo_order is None:
            raise RuntimeError("Call load_data() first.")

        result = []
        for gid in self._topo_order:
            meta = self._gid_to_meta[gid]
            result.append({
                'graph_id': gid,
                'file_path': meta['file_path'],
                'level': meta['level'],
                'parent': meta['parent'],
                'cluster_id': meta['cluster_id'],
                'is_leaf': meta['is_leaf'],
                'children': self._children_map.get(gid, []),
            })
        return result

    def get_file_path(self, graph_id: str) -> str:
        return self._gid_to_meta[graph_id]['file_path']