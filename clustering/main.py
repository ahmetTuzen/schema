import logging

from utils import set_seed
from argument_parser import parse_args

from data_loader import load_dataset # 1. dataset
from partitioning import partition # 2. leaf partition
from supernode import build as build_supernode_graph # 3. supernode graph
from supernode import report as supernode_report # 4. sparsity report
from upper_merge import build_upper_tree # 5. upper tree
from artifacts_driver import compute_all_artifacts
from serialize import serialize

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    args = parse_args()
    set_seed(args.seed, deterministic=args.deterministic)

    # 1. dataset
    data, num_nodes, edge_index, edge_weight = load_dataset(args.dataset, args.root_dir, largest_cc=args.largest_cc)

    # 2. leaf partition
    leaf = partition(
        method = args.clustering_method, 
        edge_index = edge_index,
        num_nodes = num_nodes, 
        edge_weight = edge_weight,
        objective = "modularity",
        seed = args.seed,
        max_leaf_size = args.max_leaf_size,
        )

    # 3. supernode graph
    sg = build_supernode_graph(
        edge_index = edge_index, 
        labels = leaf.labels,
        K = leaf.K, 
        edge_weight = edge_weight,
        )

    # 4. sparsity report
    stats = supernode_report(sg)

    # 5. hierarchical tree
    tree = build_upper_tree(
        sg, stats, 
        branching_cap = args.branching_cap, 
        seed = args.seed
        )

    n_internal = sum(1 for n in tree.values() if not n.is_leaf)
    n_leaves = sum(1 for n in tree.values() if n.is_leaf)
    max_level = max(n.level for n in tree.values())

    logger.info(f"Hierarchical tree built with {len(tree)} super nodes total ({n_internal}internal, "
                f"{n_leaves} leaves), depth={max_level}, root branching={len(tree[0].children)}")

    # 6. compute S / x_pool / A_pool for every internal cluster
    compute_all_artifacts(
        tree = tree,
        leaf_labels = leaf.labels,
        edge_index = edge_index,
        x = data.x,
        edge_weight = edge_weight,
        n_steps = args.propagation_steps,
        alpha = args.propagation_alpha,
    )

    # 7. serialize everything to disk
    out_dir = f"{args.output_dir}/clusters/{args.dataset}"
    if getattr(args, "run_id", None):
        out_dir = f"{out_dir}/{args.run_id}"
        
    serialize(
        tree = tree,
        leaf_labels = leaf.labels,
        x = data.x,
        edge_index = edge_index,
        edge_weight = edge_weight,
        out_dir = out_dir,
        dataset_name = args.dataset.lower(),
    )


if __name__ == "__main__":
    main()