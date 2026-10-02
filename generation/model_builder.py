import logging
logger = logging.getLogger(__name__)


def _build_node_generator(args, dataloader, max_k: int):
    node_model = getattr(args, 'node_generator', 'transformer').lower()
    if node_model == 'transformer':
        from node_generator import TransformerNodeGenerator
        cls = TransformerNodeGenerator
    else:
        raise ValueError(f"Unknown node generator '{node_model}', please implement it in generation/node_generator.py")
    return cls(args=args, dataloader=dataloader, max_k=max_k)


def _build_intra_edge_model(args, feat_dim: int):
    intra_model = getattr(args, 'intra_edge_generator', 'vgae').lower()
    if intra_model == 'vgae':
        from intra_edge_generator import VGAEEdgeGenerator
        cls = VGAEEdgeGenerator
    else:
        raise ValueError(f"Unknown intra-edge generator '{intra_model}', please implement it in generation/intra_edge_generator.py")
    return cls(args=args, feat_dim=feat_dim)

def _build_inter_edge_model(args, feat_dim: int, pool_dim: int, K: int):
    inter_model = getattr(args, 'inter_edge_generator', 'bilinear').lower()
    if inter_model == 'bilinear':
        from inter_edge_generator import SBilinearInterEdgeScorer
        cls = SBilinearInterEdgeScorer
    else:
        raise ValueError(f"Unknown inter-edge generator '{inter_model}', please implement it in generation/inter_edge_generator.py")
    return cls(args=args, feat_dim=feat_dim, pool_dim=pool_dim, K=K)