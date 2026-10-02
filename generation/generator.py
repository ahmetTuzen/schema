from graph_assembler import assemble_version
from validator import validate_v1
import torch

import logging
logger = logging.getLogger(__name__)


def run_generation(args, node_generator, intra_model, inter_model, node_dataloader, inter_loader, x_mean, x_std, ):

    common_kwargs = dict(
        node_generator = node_generator,
        intra_edge_model = intra_model,
        inter_edge_model = inter_model,
        node_dataloader = node_dataloader,
        inter_loader = inter_loader,
        x_mean = x_mean,
        x_std = x_std,
        device = args.device,
    )

    # parsing which versions to generate (1-8) from comma-separated string
    versions_str = getattr(args, 'versions', '1,8')
    try:
        versions = [int(v.strip()) for v in versions_str.split(',') if v.strip()]
    except ValueError:
        raise ValueError(f"Invalid --versions string: {versions_str!r}")
    for v in versions:
        if v not in range(1, 9):
            raise ValueError(f"Version {v} out of range (must be 1-8)")

    num_gen = int(getattr(args, 'num_generated', 10))

    logger.info(f"Generating versions {versions}, {num_gen} sample(s) each")

    #  v1: always run once for validation (checking if edge count is within the range)
    if 1 in versions:
        v1 = assemble_version(1, **common_kwargs, inter_source='file')
        torch.save(v1, f"{args.out_dir}/{args.dataset}_v1.pt")
        logger.info(f"Saved v1 to {args.out_dir}/{args.dataset}_v1.pt")
        validate_v1(v1, node_dataloader, inter_loader)
        

    for v in versions:
        if v == 1:
            continue # already handled above

        kwargs = dict(common_kwargs)
        if v in {2, 3, 4}:
            kwargs['inter_source'] = 'file'

        for i in range(num_gen):
            sample = assemble_version(v, **kwargs)
            out_path = f"{args.out_dir}/{args.dataset}_v{v}_{i}.pt"
            torch.save(sample, out_path)
            logger.info(f"Saved v{v} sample {i} to {out_path}")