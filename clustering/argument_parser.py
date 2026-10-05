import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Scalable Hierarchical Graph Generation via Soft Community Structure - Clustering Stage",                    
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Dataset related settings
    data_parser = parser.add_argument_group("Dataset")
    data_parser.add_argument("--dataset", type=str, default="citeseer", help="Dataset to use (Please check data_loader.py)")
    data_parser.add_argument("--root_dir", type=str, default="./data", help="Root directory for datasets")

    # Clustering related settings
    clustering_parser = parser.add_argument_group("Clustering")
    clustering_parser.add_argument("--clustering_method", type=str, default="leiden", choices=["leiden", "louvain"], help="Clustering method to use")
    clustering_parser.add_argument("--branching_cap", type=int, default=16, help="Maximum branching factor for building upper tree.")
    clustering_parser.add_argument("--max_leaf_size", type=int, default=0, help="Split any leaf cluster larger than this by re-running community "
                                   "detection on its induced subgraph. 0 to disable (In the paper, only used for IGB Medium)")
    clustering_parser.add_argument("--largest_cc", action="store_true", help="Proceed with largest connected component")

    # Soft assignment settings
    soft_parser = parser.add_argument_group("Soft Assignment")
    soft_parser.add_argument("--propagation_steps", type=int, default=3, help="Propagation steps (Eq. 1. tau)")
    soft_parser.add_argument("--propagation_alpha", type=float, default=0.7, help="Propagation coefficient (Eq. 1. alpha)")

    # Output settings
    output_parser = parser.add_argument_group("Output")
    output_parser.add_argument("--output_dir", type=str, default="./output", help="Directory to save output files")
    output_parser.add_argument("--run_id", type=str, default=None, help="Sub-dir name for experimenting {output_dir}/clusters/{dataset}/{run_id}")

    # Reproducibility settings
    reproducibility_parser = parser.add_argument_group("Reproducibility")
    reproducibility_parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    reproducibility_parser.add_argument("--deterministic", action="store_true", default=False, help="Enable strict CUDA determinism (way slower)")

    parser.set_defaults(notify=False)

    return parser.parse_args()