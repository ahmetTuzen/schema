import argparse
import os


DATASET_DEFAULTS = {
    # Dataset defaults can be overridden by CLI flags, but it is safe to add here. 
    "citeseer": {"directed": False, "weighted": False,},
    "cora_ml": {"directed": False, "weighted": False},
    "amazon_photo" : {"directed": False, "weighted": False},
    "amazon_computers" : {"directed": False, "weighted": False},
    "flickr": {"directed": False, "weighted": False},
    "reddit": {"directed": False, "weighted": False},
    "yelp": {"directed": False, "weighted": False},
    "dgraphfin": {"directed": True, "weighted": False},
    "ogbn_products": {"directed": False, "weighted": False},
    "igb": {"directed": False, "weighted": False},
    }

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Scalable Hierarchical Graph Generation via Soft Community Structure - Training and Generation Stage",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Dataset related settings
    data_parser = parser.add_argument_group("Dataset")
    data_parser.add_argument("--dataset", type=str, default="citeseer", help="Dataset name (case-insensitive for dataset-specific defaults)")
    data_parser.add_argument("--data", type=str, default="data", help="Path to the data directory")
    data_parser.add_argument("--json-path", type=str, default=None, help="Path to cluster_mapping.json.")

    data_parser.add_argument("--directed", action="store_true", default=None, help="Treat edges as directed (default depends on dataset)")
    data_parser.add_argument("--undirected", dest="directed", action="store_false", help="Treat edges as undirected")
    data_parser.add_argument("--weighted", action="store_true", default=None, help="Use edge weights if available (default depends on dataset)")
    data_parser.add_argument("--unweighted", dest="weighted", action="store_false", help="Ignore edge weights, even if present in the data")

    # Shared architecture settings
    shared_arch_parser = parser.add_argument_group("Shared architecture")
    shared_arch_parser.add_argument("--latent-dim", type=int, default=64, help="Hidden dim for node generator")
    shared_arch_parser.add_argument("--edge-emb-dim", type=int, default=128, help="Node embedding dim inside edge generators")
    shared_arch_parser.add_argument("--edge-hidden-dim", type=int, default=256, help="Hidden dim for pair-scoring MLPs")
    shared_arch_parser.add_argument("--dropout", type=float, default=0.1)

    # Node generation settings
    node_parser = parser.add_argument_group("Node generation")
    node_parser.add_argument("--num-layers", type=int, default=2, help="Depth of node-generator backbone")
    node_parser.add_argument("--nhead", type=int, default=4, help="Attention heads for transformer")
    node_parser.add_argument("--noise-during-training", action="store_true", help="Inject noise in forward() so noise_proj is trained")
    node_parser.add_argument("--train-noise", type=float, default=1.0, help="Noise used during training when noise_during_training is set")
    node_parser.add_argument("--node-loss", type=str, default="huber", choices=["mse", "mae", "huber", "cosine", "combined"], help="Reconstruction loss for the node generator")
    node_parser.add_argument("--huber-delta", type=float, default=1.0, help="Delta parameter for Huber / combined loss")
    node_parser.add_argument("--cosine-weight", type=float, default=0.1, help="Weight of the cosine term in the combined loss")
    node_parser.add_argument("--feature-type", type=str, default="auto", choices=["auto", "continuous", "binary"], help="Reference feature type.")

    # Intra-edge generation settings
    intra_parser = parser.add_argument_group("Intra-edge generation")
    intra_parser.add_argument("--neg-ratio", type=float, default=1.0, help="Negative samples per positive pair during training")
    intra_parser.add_argument("--weight-coef", type=float, default=0.1, help="Weight of the edge-weight Huber loss term")
    intra_parser.add_argument("--gen-chunk-size", type=int, default=256, help="Row chunk size for N^2 score/weight materialization at generation")
    intra_parser.add_argument("--use-generated-x", action="store_true", default=False, help="Feed previously-generated node features to the edge model")
    intra_parser.add_argument("--intra-max-pairs", type=int, default=10000, help="Max positive pairs sampled per leaf per epoch in intra-edge training; 0 disables the cap. (GPU Memory)")
    intra_parser.add_argument("--intra-posterior-max-edges", type=int, default=1_000_000, help="Max edges fed to the VGAE posterior encoder at generation; 0 disables the cap.")
    
    # Inter-edge generation settings
    inter_parser = parser.add_argument_group("Inter-edge generation")
    inter_parser.add_argument("--inter-prior-mode", type=str, default="fusion", choices=["none", "kl", "fusion"], help="Affinity from hierarchy")
    inter_parser.add_argument("--prior-weight", type=float, default=0.5, help="Initial value for the learnable prior alpha")
    inter_parser.add_argument("--prior-kl-coef", type=float, default=0.1, help="Scale of the KL prior-alignment regularizer")
    inter_parser.add_argument("--s-threshold", type=float, default=0.1, help="Minimum S-membership for candidate filtering")
    inter_parser.add_argument("--topk-nodes", type=int, default=500, help="Top-k cap on candidate filtering per cluster")
    inter_parser.add_argument("--inter-gen-cluster-threshold", type=float, default=0.05, help="Minimum S-membership for candidate filtering at generation time")
    inter_parser.add_argument("--inter-gen-edge-threshold", type=float, default=0.8, help="Minimum score for edge generation at generation time")
    inter_parser.add_argument("--inter-child-agg", type=str, default="mean", choices=["mean", "sum", "attention"], help="Aggregation over child clusters")
    inter_parser.add_argument("--inter-max-pairs", type=int, default=10000, help="Max positive inter-edge pairs sampled per subgraph per epoch; 0 disables the cap.")
    inter_parser.add_argument("--inter-gnn-max-edges", type=int, default=2_000_000, help="Max intra edges fed to the GNN refinement at generation; 0 disables the cap.")
    inter_parser.add_argument("--inter-neg-pool-frac", type=float, default=0.01, help="Negatives drawn per subgraph as a fraction of the candidate pool, so negative density does not fall as topk_nodes grows.")
    inter_parser.add_argument("--inter-max-neg", type=int, default=20000, help="Hard cap on negatives per subgraph.")
    inter_parser.add_argument("--inter-budget-scale", type=float, default=1.0, help="Multiplier on the A_pool ceiling for how many crossings a cluster pair may emit. 0 disables the ceiling.")
    
    # Training settings
    training_parser = parser.add_argument_group("Training")
    training_parser.add_argument("--batch-size", type=int, default=16)
    training_parser.add_argument("--learning-rate", type=float, default=1e-3)
    training_parser.add_argument("--num-epochs", type=int, default=25, help="Default epoch count used for any stage not given its own override")
    training_parser.add_argument("--node-epochs", type=int, default=None, help="Epochs for the node stage")
    training_parser.add_argument("--intra-epochs", type=int, default=None, help="Epochs for the intra-edge stage (defaults to --num-epochs)")
    training_parser.add_argument("--inter-epochs", type=int, default=None, help="Epochs for the inter-edge stage (defaults to --num-epochs)")

    # Generation settings
    gen_parser = parser.add_argument_group("Generation")
    gen_parser.add_argument("--num-generated", type=int, default=10, help="Number of samples to generate per version")
    gen_parser.add_argument("--versions", type=str, default="1,8", help="Comma-separated list of versions to generate, please check ablation study. best to leave at 1,8")
    gen_parser.add_argument("--denormalize-nodes", action="store_true", default=False, help="Flag for denormalizing nodes at the generation stage")
    gen_parser.add_argument("--out-dir", type=str, default=None, help="Dir for checkpoints + generated samples. Defaults to --data.")

    # Runtime and reproducibility settings
    run_parser = parser.add_argument_group("Runtime and reproducibility")
    run_parser.add_argument("--device", type=str, default="cuda", help="Device to run")
    run_parser.add_argument("--skip-training", action="store_true", default=False, help="Load checkpoints when present instead of training")
    run_parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    run_parser.add_argument("--deterministic", action="store_true", default=False, help="Enable strict CUDA determinism (way slower)")

    return parser

def _apply_dataset_defaults(args: argparse.Namespace) -> None:
    defaults = DATASET_DEFAULTS.get(args.dataset.lower(), {})

    if args.directed is None:
        args.directed = defaults.get("directed", False)
    if args.weighted is None:
        args.weighted = defaults.get("weighted", False)


def _resolve_per_stage_epochs(args: argparse.Namespace) -> None:
    """If node epochs, or edge epochs are not passed, set them to num_epoch"""
    for attr in ("node_epochs", "intra_epochs", "inter_epochs"):
        if getattr(args, attr) is None:
            setattr(args, attr, args.num_epochs)


def _resolve_json_path(args: argparse.Namespace) -> None:
    if args.json_path is None:
        args.json_path = f"output/clusters/{args.dataset}/cluster_mapping.json"


def _add_compat_aliases(args: argparse.Namespace) -> None:
    args.node_generator = "transformer"
    args.intra_edge_generator = "vgae"
    args.inter_edge_generator = "bilinear"


def _resolve_out_dir(args):
    if getattr(args, "out_dir", None) is None:
        args.out_dir = "output"
    os.makedirs(args.out_dir, exist_ok=True)

def parse_args() -> argparse.Namespace:
    # To make sure everything will work without any error.
    parser = _build_parser()
    args = parser.parse_args()

    _apply_dataset_defaults(args)
    _resolve_per_stage_epochs(args)
    _resolve_json_path(args)
    _add_compat_aliases(args)
    _resolve_out_dir(args)

    return args