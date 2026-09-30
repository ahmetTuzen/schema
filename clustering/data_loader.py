import logging
import torch
import os

logger = logging.getLogger(__name__)


def load_dataset(name: str, root_dir: str, largest_cc: bool = False):
    """
    Returns
    -------
    data : the raw PyG Data object
    num_nodes : int
    edge_index : [2, E] LongTensor
    """
    dataset = name.lower()

    if dataset == "citeseer":
        from torch_geometric.datasets import Planetoid
        data = Planetoid(root=root_dir, name="CiteSeer")[0]
    elif dataset == "cora_ml":
        from torch_geometric.datasets import CitationFull
        data = CitationFull(root=root_dir, name="Cora_ML")[0]
    elif dataset == "amazon_photo":
        from torch_geometric.datasets import Amazon
        data = Amazon(root=root_dir, name="Photo")[0]
    elif dataset == "amazon_computers":
        from torch_geometric.datasets import Amazon
        data = Amazon(root=root_dir, name="Computers")[0]

    elif dataset == "flickr":
        from torch_geometric.datasets import Flickr
        data = Flickr(root=f"{root_dir}/Flickr")[0]

    elif dataset == "reddit":
        from torch_geometric.datasets import Reddit
        data = Reddit(root=f"{root_dir}/Reddit")[0]
    elif dataset == "yelp":
        from torch_geometric.datasets import Yelp
        data = Yelp(root=f"{root_dir}/Yelp")[0]
    elif dataset == "dgraphfin":
        from torch_geometric.datasets import DGraphFin
        data = DGraphFin(root=f"{root_dir}/DGraphFin")[0]
    elif dataset in ("ogbn_products", "ogbn", "ogbn-products", "products"):
        from ogb.nodeproppred import PygNodePropPredDataset
        data = PygNodePropPredDataset(name="ogbn-products", root=f"{root_dir}/ogb")[0]
    elif dataset.startswith("igb_"):
        import glob
        import numpy as np
        from torch_geometric.data import Data

        size = "medium"
        feat_dim = int(os.environ.get("IGB_FEAT_DIM", 128))
        hits = glob.glob(os.path.join(root_dir, "**", size, "processed", "paper", "node_feat.npy"), recursive=True)
        paper_dir = os.path.dirname(hits[0])
        base = os.path.dirname(paper_dir)
        ei_path = os.path.join(base, "paper__cites__paper", "edge_index.npy")
        if not os.path.exists(ei_path):
            cand = glob.glob(os.path.join(base, "**", "edge_index.npy"), recursive=True)
            if not cand:
                raise FileNotFoundError(f"edge_index.npy not found under {base}")
            ei_path = cand[0]

        ei_np = np.load(ei_path, mmap_mode="r")
        feat_np = np.load(os.path.join(paper_dir, "node_feat.npy"), mmap_mode="r")
        logger.info(f"IGB {size}: edge_index {ei_np.shape} {ei_np.dtype}, node_feat {feat_np.shape} {feat_np.dtype}")

        edge_index = torch.from_numpy(np.ascontiguousarray(ei_np)).long()
        if edge_index.shape[0] != 2:
            edge_index = edge_index.t().contiguous()
        del ei_np

        N, F_full = feat_np.shape
        D = min(feat_dim, F_full)
        x = torch.empty(N, D, dtype=torch.float32)
        CH = 200_000 # rows per chunck, cant put everything to memory
        for i in range(0, N, CH):
            x[i:i + CH] = torch.from_numpy(np.asarray(feat_np[i:i + CH, :D], dtype=np.float32))
        del feat_np
        logger.info(f"IGB {size}: kept first {D} of {F_full} feature dims -> x {tuple(x.shape)}")

        data = Data(x=x, edge_index=edge_index)
        data.num_nodes = N       
    else:
        raise ValueError(f"Unsupported dataset: {name}.\nAdd a branch in ./clustering/data_loader.load_dataset to support it.")
    
    if largest_cc:
        from torch_geometric.transforms import LargestConnectedComponents
        before = data.num_nodes
        data = LargestConnectedComponents()(data)
        logger.info(f"largest_cc: {before} -> {data.num_nodes} nodes")

    num_nodes = data.num_nodes
    edge_index = data.edge_index

    edge_weight = getattr(data, "edge_attr", None)
    if edge_weight is not None and edge_weight.dim() > 1:
        logger.info(f"edge_attr has shape {tuple(edge_weight.shape)}; ignoring as multi-dimensional is not supported.")
        edge_weight = None

    logger.info(f"Loaded {name}: {num_nodes} nodes, {data.num_edges} edges, weighted={edge_weight is not None}")

    return data, num_nodes, edge_index, edge_weight