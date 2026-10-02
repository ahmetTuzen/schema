import torch

import logging
logger = logging.getLogger(__name__)

def _collect_ground_truth_stats(node_dataloader, inter_loader) -> dict:
    """ Rebuild the input graph's stats from leaf + subgraph .pt files. """
    dataset = node_dataloader.dataset
    intra_edges = set()
    node_ids = set()

    for path in dataset.file_paths:
        leaf = torch.load(path, weights_only=False)
        orig = leaf.original_node_indices
        for g in orig.tolist():
            node_ids.add(g)
        src, dst = leaf.edge_index[0].tolist(), leaf.edge_index[1].tolist()
        for s, d in zip(src, dst):
            intra_edges.add((min(orig[s].item(), orig[d].item()), max(orig[s].item(), orig[d].item())))

    inter_edges = set()
    for node in inter_loader.bottom_up_order():
        data = torch.load(node['file_path'], weights_only=False)
        orig = data.original_node_indices if hasattr(data, 'original_node_indices') else None

        if hasattr(data, 'inter_local_node'):
            loc_t = data.inter_local_node.long()
            ext_t = data.inter_external.long()
            if loc_t.numel() == 0:
                continue
            src_g_t = orig[loc_t].long() if orig is not None else loc_t
            lo = torch.minimum(src_g_t, ext_t).tolist()
            hi = torch.maximum(src_g_t, ext_t).tolist()
            inter_edges.update(zip(lo, hi))
            continue

        ies = getattr(data, 'inter_cluster_edges', None) or []
        if not ies:
            continue
        sample = ies[0]
        for e in ies:
            if 'local_node' in sample:
                src_g = int(orig[int(e['local_node'])].item()) if orig is not None else int(e['local_node'])
                dst_g = int(e['external_node'])
            elif 'src' in sample:
                if e.get('cluster_src', -1) != node['cluster_id']:
                    continue
                src_g = int(e['src'])
                dst_g = int(e['dst'])
            else:
                continue
            inter_edges.add((min(src_g, dst_g), max(src_g, dst_g)))

    return {
            'num_nodes': len(node_ids),
            'num_intra_edges': len(intra_edges),
            'num_inter_edges': len(inter_edges),
            'total_edges': len(intra_edges) + len(inter_edges),
        }


def validate_v1(v1_data, node_dataloader, inter_loader, node_tol: float = 0.0, edge_tol: float = 0.05) -> bool:
    """
    Validate the v1 graph against the ground truth stats. Returns True if within tolerance, False otherwise. 
    Even if the validation fails, the generation continues, but logs warnings.
    """
    truth = _collect_ground_truth_stats(node_dataloader, inter_loader)

    got_nodes = int(v1_data.num_nodes)
    src, dst = v1_data.edge_index[0], v1_data.edge_index[1]
    canon = torch.stack([torch.minimum(src, dst), torch.maximum(src, dst)], dim=0)
    got_edges_undirected = int(torch.unique(canon.t(), dim=0).shape[0])
    got_edges_directed = int(v1_data.edge_index.shape[1])

    logger.info(f"[validate v1] ground truth: {truth}")
    logger.info(f"[validate v1] v1: nodes={got_nodes}, edges_directed={got_edges_directed}, edges_undirected={got_edges_undirected}")

    node_ok = abs(got_nodes - truth['num_nodes']) <= max(1, int(truth['num_nodes'] * node_tol))

    truth_total = truth['total_edges']
    edge_ok = abs(got_edges_undirected - truth_total) <= max(1, int(truth_total * edge_tol))

    if node_ok and edge_ok:
        logger.info("[validate v1] PASS")
        return True

    problems = []
    if not node_ok:
        problems.append(f"node count mismatch (got {got_nodes}, truth {truth['num_nodes']})")
    if not edge_ok:
        problems.append(f"edge count out of tolerance (got {got_edges_undirected} unique undirected pairs, truth {truth_total})")

    logger.warning(f"[validate v1] FAIL: {'; '.join(problems)}")
    return False
