import os
import torch
from tabulate import tabulate

import igraph as ig
import numpy as np
import scipy.sparse as sp
import networkx as nx

from torch_geometric.seed import seed_everything


def set_seed(seed: int, deterministic: bool = False) -> None:
    """
    Seed all RNGs via PyG's seed_everything (covers random, numpy, torch, torch.cuda). 
    Optional deterministic mode enables determinism at a significant performance cost.
    """

    seed_everything(seed)

    if deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)

def _undirected(g):
    return g.as_undirected(mode="collapse") if g.is_directed() else g


def _adjacency(g):
    """0/1 symmetric scipy CSR adjacency of the undirected version. Implemented it for motif count."""
    gu = _undirected(g)
    n = gu.vcount()
    el = gu.get_edgelist()
    if not el:
        return sp.csr_matrix((n, n))
    src, dst = zip(*el)
    rows = np.concatenate([src, dst])
    cols = np.concatenate([dst, src])
    A = sp.csr_matrix((np.ones(rows.shape[0]), (rows, cols)), shape=(n, n))
    A.data[:] = 1.0
    return A



def data_to_igraph(data, directed=True, simplify=True):
    """PyG Data -> igraph.Graph. Uses edge_index only"""
    ei = data.edge_index.cpu().numpy()
    n = int(data.num_nodes) if getattr(data, "num_nodes", None) is not None else (int(ei.max()) + 1 if ei.size else 0)
    edges = list(zip(ei[0].tolist(), ei[1].tolist())) if ei.size else []
    g = ig.Graph(n=n, edges=edges, directed=directed == "directed")
    if simplify:
        g.simplify(multiple=True, loops=True)
    return g

def data_to_nx(data):
    A = _adjacency(data)
    A.setdiag(0)
    A.eliminate_zeros()
    return nx.from_scipy_sparse_array(A)

def load_baselines(cfg):
    models = {}

    """ If there are another format, add it here. We just converted everything to pt format."""
    if cfg.dataset.storage.format == "torch":
        g_ref = torch.load(cfg.dataset.storage.path, weights_only=False)
        models["Original"] = g_ref
    
    for model in cfg.method:
        if model.format == "torch":
            try:
                models[model.name] = torch.load(model.output, weights_only=False)
            except Exception:
                continue
    
    return models

def report_fidelity(df, path):
    table_output = tabulate(df, headers="keys", tablefmt="github", floatfmt=".4f")
    print(table_output)

    if path.endswith('.csv'):
        txt_path = path[:-4] + '.txt'

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(table_output)