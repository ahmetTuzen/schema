import logging

from argument_parser import parse_args

from trainer import node_training, intra_edge_training, inter_edge_training, save_reconstructed_x
from generator import run_generation
from utils import set_seed, stage_metrics

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO)


def main(args):
    logger.info(f"Running experiment with dataset: {args.dataset}")
    skip = bool(getattr(args, 'skip_training', False))
    
    with stage_metrics('node_train'):
        node_generator, node_dataloader, x_mean, x_std = node_training(args, skip_training=skip)
    if getattr(args, 'use_generated_x', False):
        save_reconstructed_x(args, node_generator, node_dataloader, x_mean, x_std)
    with stage_metrics('intra_train'):
        intra_model = intra_edge_training(args, x_mean, x_std, skip_training=skip)
    with stage_metrics('inter_train'):
        inter_model, inter_loader = inter_edge_training(args, x_mean, x_std, skip_training=skip)

    with stage_metrics('generation'):
        run_generation(
            args=args,
            node_generator=node_generator,
            intra_model=intra_model,
            inter_model=inter_model,
            node_dataloader=node_dataloader,
            inter_loader=inter_loader,
            x_mean=x_mean,
            x_std=x_std,
        )


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed, deterministic=args.deterministic)
    
    root = logging.getLogger()
    root.setLevel(logging.INFO)


    main(args)