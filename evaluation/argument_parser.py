import argparse

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Scalable Hierarchical Graph Generation via Soft Community Structure - Evaluation Stage",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--evaluation-stage", type=str, default="structural", choices=["all", "structural", "memorization", "downstream"], 
                        help="Evaluation stage from the paper")
    parser.add_argument("--evaluation-mode", type=str, default="calculate", choices=["calculate", "report", "append"], 
                        help="Evaluation mode, calculate: evaluates from scratch; report: prints and plots from csv; append: extend results to csv file.")
    parser.add_argument("--yaml-location", type=str, default="report/yaml/citeseer.yaml", help="Location of the yaml file for evaluation configuration")
    parser.add_argument("--csv-location", type=str, default="report", help="Output directory for evaluation results in csv format")

    parser.add_argument("--device", type=str, default="cuda", help="Device to run (cuda only helpful for downstream evaluation)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument("--deterministic", action="store_true", default=False, help="Enable strict CUDA determinism (way slower)")

    return parser


def parse_args() -> argparse.Namespace:
    parser = _build_parser()
    args = parser.parse_args()

    return args