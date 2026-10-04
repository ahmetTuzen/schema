import os

import pandas as pd
import numpy as np

from utils import report_fidelity

import logging
logger = logging.getLogger(__name__)

TOPOLOGY_STRATEGY = {
    "small": ["avg_degree", "deg_dist_w1", "assort", "power_law", 
                "clus_coef", "triangle", "square", "orbit",
                "cpl", "lap_spec_w1",
                "modularity", "comm_dist_w1", "inter_ratio"],
    "medium": ["avg_degree", "deg_dist_w1", "assort", "power_law", 
                "clus_coef", "triangle", "square",
                "modularity", "comm_dist_w1", "inter_ratio"],
    "big": ["avg_degree", "deg_dist_w1", "assort", "power_law", 
                "modularity", "comm_dist_w1", "inter_ratio"],
    "paper_appendix": ["avg_degree", "deg_dist_w1", "triangle", "clus_coef", 
                "assort", "modularity", "inter_ratio", "cpl",],
    "paper_main": ["deg_dist_w1", "triangle", "assort", "inter_ratio"]
    }


def structural_fidelity(args, cfg):
    size = cfg.dataset.graph.scale
    if args.evaluation_mode == "report":
        logger.info("Printing topological fidelity report from existing results...")
        file_path = os.path.join(args.csv_location, "topological_fidelity", cfg.dataset.name + ".csv")
        topological_df = pd.read_csv(file_path)  

    elif args.evaluation_mode == "calculate" or args.evaluation_mode == "append":
        logger.info("Calculating topological fidelity...")
        logger.info(f"Dataset size is {size}; {len(TOPOLOGY_STRATEGY[size])} different evaluation metrics will be computed (additional metrics may be included in the report).")

        topological_fidelity_results = calculate_topological_fidelity(cfg)
        logger.info("Topological fidelity evaluation is completed. Saving .csv file, then reporting...")

        topological_df, save_path = save_topological_fidelity(args, cfg, topological_fidelity_results)
        
    report_fidelity(topological_df, save_path)


def calculate_topological_fidelity(cfg):
    topology_evaluations = TOPOLOGY_STRATEGY[cfg.dataset.graph.scale]

    from utils import load_baselines, data_to_igraph

    baselines = load_baselines(cfg)
    baselines = {name: data_to_igraph(data, directed=cfg.dataset.graph.mode) for name, data in baselines.items()}

    leidens = {}
    results = {name: {} for name in baselines}

    # TODO: implement metric for directed version as well. Currently, all metrics are for undirected graphs.
    # But this should be fine as we also converting the reference graph to undirected for the evaluation.

    if "avg_degree" in topology_evaluations:
        from topological_metrics import avg_degree

        ref_val = avg_degree(baselines["Original"])

        for name, g in baselines.items():
            val = avg_degree(g)
            results[name]["avg_degree"] = val
            results[name]["avg_degree_ratio"] = val / ref_val

    if "deg_dist_w1" in topology_evaluations:
        from topological_metrics import w1_distance, degree_sequence

        ref = degree_sequence(baselines["Original"])
        results["Original"]["deg_dist_w1"] = 0

        for name, g in baselines.items():
            val = degree_sequence(g)
            results[name]["deg_dist_w1"] = w1_distance(val, ref)

    if "assort" in topology_evaluations:
        from topological_metrics import assortativity

        ref_val = assortativity(baselines["Original"])
        results["Original"]["assortativity"] = 0

        for name, g in baselines.items():
            val = assortativity(g)
            results[name]["assortativity"] = val
            results[name]["assortativity_diff"] = abs(val-ref_val)

    if "power_law" in topology_evaluations:
        from topological_metrics import power_law_exponent

        ref_val = power_law_exponent(baselines["Original"])
        results["Original"]["power_law"] = 0

        for name, g in baselines.items():
            val = power_law_exponent(g)
            results[name]["power_law"] = val
            results[name]["power_law_diff"] = abs(val-ref_val)

    if "clus_coef" in topology_evaluations:
        from topological_metrics import clustering_coefficient

        ref_val = clustering_coefficient(baselines["Original"])
        results["Original"]["clus_coef"] = 0

        for name, g in baselines.items():
            val = clustering_coefficient(g)
            results[name]["clus_coef"] = val
            results[name]["clus_coef_ratio"] = val / ref_val

    if "triangle" in topology_evaluations:
        from topological_metrics import triangle_count

        ref_val = triangle_count(baselines["Original"])
        results["Original"]["triangle"] = 0

        for name, g in baselines.items():
            val = triangle_count(g)
            results[name]["triangle"] = val
            results[name]["triangle_ratio"] = val / ref_val

    if "square" in topology_evaluations:
        from topological_metrics import square_count

        ref_val = square_count(baselines["Original"])
        results["Original"]["square"] = 0

        for name, g in baselines.items():
            val = square_count(g)
            results[name]["square"] = val
            results[name]["square_ratio"] = val / ref_val

    if "orbit" in topology_evaluations:
        from topological_metrics import w1_distance, orbit_dist
        from utils import data_to_nx

        ref = orbit_dist(data_to_nx(baselines["Original"]))
        results["Original"]["orbit_w1"] = 0

        for name, g in baselines.items():
            val = orbit_dist(data_to_nx(g))
            results[name]["orbit_w1"] = float(np.mean([w1_distance(val[:, k], ref[:, k]) for k in range(val.shape[1]) ]) )
        

    if "cpl" in topology_evaluations:
        from topological_metrics import characteristic_path_length

        ref_val = characteristic_path_length(baselines["Original"])
        results["Original"]["cpl"] = 0

        for name, g in baselines.items():
            val = characteristic_path_length(g)
            results[name]["cpl"] = val
            results[name]["cpl_ratio"] = val / ref_val

    if "lap_spec_w1" in topology_evaluations:
        from topological_metrics import w1_distance, laplacian_eigenvalues

        ref = laplacian_eigenvalues(baselines["Original"])

        for name, g in baselines.items():
            val = laplacian_eigenvalues(g)
            results[name]["lap_w1"] = w1_distance(val, ref)

    if "modularity" in topology_evaluations:
        if not leidens:
            from topological_metrics import leiden
            for name, g in baselines.items():
                leidens[name] = leiden(g)

        from topological_metrics import modularity

        ref_val = modularity(leidens["Original"])

        for name, comm in leidens.items():
            val = modularity(comm)
            results[name]["modularity"] = val
            results[name]["modularity_diff"] = abs(val-ref_val)


    if "comm_dist_w1" in topology_evaluations:
        if not leidens:
            from topological_metrics import leiden
            for name, g in baselines.items():
                leidens[name] = leiden(g)

        from topological_metrics import w1_distance, community_size_sequence

        ref = community_size_sequence(leidens["Original"])
        results["Original"]["comm_dist_w1"] = 0

        for name, comm in leidens.items():
            val = community_size_sequence(comm)
            results[name]["comm_dist_w1"] = w1_distance(val, ref)

    if "inter_ratio"  in topology_evaluations:
        if not leidens:
            from topological_metrics import leiden
            for name, g in baselines.items():
                leidens[name] = leiden(g)

        from topological_metrics import inter_intra_ratio

        ref_val = inter_intra_ratio(leidens["Original"])
        results["Original"]["inter_intra"] = 0

        for name, comm in leidens.items():
            val = inter_intra_ratio(comm)
            results[name]["inter_intra"] = val
            results[name]["inter_intra_ratio"] = val / ref_val

    return results





def save_topological_fidelity(args, cfg, results):
    save_path = os.path.join(args.csv_location, "topological_fidelity", cfg.dataset.name + ".csv")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    df = pd.DataFrame.from_dict(results, orient="index")
    df.index.name = "model"
    df = df.reset_index()

    if args.evaluation_mode == "append":
        df_old = pd.read_csv(save_path)
        df_new = df

        df = pd.concat([df_old, df_new], ignore_index=True)
        df = df.drop_duplicates(subset=["model"], keep="last")

    df.to_csv(save_path, index=False)

    return df, save_path

