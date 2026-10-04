from itertools import combinations

import numpy as np


def _edge_set(edge_index):
    """Undirected, and dropped duplicates and self-loops."""
    ei = np.asarray(edge_index)
    a = np.minimum(ei[0], ei[1])
    b = np.maximum(ei[0], ei[1])
    m = a != b
    return set(zip(a[m].tolist(), b[m].tolist()))


def _neighbours(edge_index, num_nodes):
    nbr = [set() for _ in range(num_nodes)]
    ei = np.asarray(edge_index)
    for u, v in zip(ei[0].tolist(), ei[1].tolist()):
        if u != v:
            nbr[u].add(v)
            nbr[v].add(u)
    return nbr


def _num_nodes(*edge_indices):
    """If no num_nodes is given, infer it from the edge_index arrays."""
    return int(max(np.asarray(e).max() for e in edge_indices)) + 1



def edge_overlap(real_ei, gen_ei):
    """
    This explained in the paper. 
    Simply jaccard similarity of the generated edges vs the reference edges.
    """
    R, G = _edge_set(real_ei), _edge_set(gen_ei)
    I = R & G
    return len(I) / max(len(R | G), 1)


def self_overlap(gen_eis, return_std=False):
    """
    Also expained in the paper.
    Overlap between generated samples. Required to generate more than 1 otherwise not defined.
    """
    pairs = list(combinations(range(len(gen_eis)), 2))
    if not pairs:
        return (None, None) if return_std else None
    vals = [edge_overlap(gen_eis[i], gen_eis[j])["jaccard"] for i, j in pairs]
    if return_std:
        std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        return float(np.mean(vals)), std
    return float(np.mean(vals))


def neighborhood_memorization(real_ei, gen_ei, num_nodes=None, threshold=0.9):
    """
    Also expained in the paper.

    Nodes that are isolated in BOTH graphs are excluded from the average.

    This reports both 'frac' in the appendix, and neighborhood memorization main and appendix.
    """
    if num_nodes is None:
        num_nodes = _num_nodes(real_ei, gen_ei)
    Rn = _neighbours(real_ei, num_nodes)
    Gn = _neighbours(gen_ei, num_nodes)

    jac = []
    for i in range(num_nodes):
        r, g = Rn[i], Gn[i]
        union = len(r | g)
        if union == 0:
            continue
        jac.append(len(r & g) / union)

    if not jac:
        return {
            "neighbour_jaccard_mean": float("nan"),
            "memorized_node_frac": float("nan"),
        }

    jac = np.asarray(jac)
    return {
        "neighbour_jaccard_mean": float(jac.mean()),
        "memorized_node_frac": float((jac >= threshold).mean()),
    }



def nndr(real_x, gen_x):
    """
    This evaluation is only in appendix. Please follow there

    Reporting nndr and leak.
    """
    from sklearn.neighbors import NearestNeighbors
    real_x = np.asarray(real_x, dtype=np.float64)
    gen_x = np.asarray(gen_x, dtype=np.float64)

    nn = NearestNeighbors(n_neighbors=2).fit(real_x)
    d_gen, _ = nn.kneighbors(gen_x) # (Ngen, 2)
    dcr = d_gen[:, 0]
    nndr = dcr / np.clip(d_gen[:, 1], 1e-12, None)

    d_real, _ = nn.kneighbors(real_x) 
    thr = np.quantile(d_real[:, 1], 0.05) 

    return {
        "nndr_median": float(np.median(nndr)),
        "leak_frac": float((dcr < thr).mean()),
    }