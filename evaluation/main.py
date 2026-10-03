from argument_parser import parse_args
from utils import set_seed
from omegaconf import OmegaConf

import logging
logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO)


def main(args, cfg):
    logger.info(f"Evaluation on the dataset {cfg.dataset.name} with {len(cfg.method)} baselines has started...")
    logger.info(f"Evaluation mode: {args.evaluation_mode}")

    if args.evaluation_stage == "all" or args.evaluation_stage == "structural":
        from structural_evaluation import structural_fidelity
        structural_fidelity(args, cfg)
    if args.evaluation_stage == "all" or args.evaluation_stage == "memorization":
        from memorization_evaluation import memorization_fidelity
        memorization_fidelity(args, cfg)
    if args.evaluation_stage == "all" or args.evaluation_stage == "downstream":
        from downstream_evaluation import downstream_fidelity
        downstream_fidelity(args, cfg)


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed, deterministic=args.deterministic)

    cfg = OmegaConf.load(args.yaml_location)

    main(args, cfg)