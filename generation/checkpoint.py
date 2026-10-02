import logging
logger = logging.getLogger(__name__)


def _node_ckpt_path(args) -> str:
    model = getattr(args, 'node_generator', 'transformer')
    logger.info(f"Node generator model: {model}")
    return f"{args.out_dir}/{args.dataset}_{model}.pt"


def _intra_ckpt_path(args) -> str:
    model = getattr(args, 'intra_edge_generator', 'cvae')
    logger.info(f"Intra edge generator model: {model}")
    return f"{args.out_dir}/{args.dataset}_{model}_intra_edge.pt"


def _inter_ckpt_path(args) -> str:
    model = getattr(args, 'inter_edge_generator', 'bilinear')
    logger.info(f"Inter edge generator model: {model}")
    return f"{args.out_dir}/{args.dataset}_{model}_inter_edge.pt"