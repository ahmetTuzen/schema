import numpy as np
from scipy.stats import wasserstein_distance
import igraph as ig
import random
from utils import _adjacency, _undirected
import orbit_count


def avg_degree(g):
    """Directed -> {'in', 'out'} means; undirected -> single mean degree."""
    if g.is_directed():
        return {"in": float(np.mean(g.indegree())), "out": float(np.mean(g.outdegree()))}
    return float(np.mean(g.degree()))


def assortativity(g):
    """Degree assortativity coefficient (undirected). NaN if all degrees equal."""
    return _undirected(g).assortativity_degree(directed=False)


def power_law_exponent(g):
    """MLE power-law exponent (alpha) of the degree distribution"""
    deg = np.array(_undirected(g).degree())
    deg = deg[deg > 0]
    return float(ig.statistics.power_law_fit(deg).alpha)


def clustering_coefficient(g, mode="global"):
    """mode='global' -> transitivity; mode='avg_local' -> average local clustering."""
    gu = _undirected(g)
    if mode == "global":
        return gu.transitivity_undirected(mode="zero")
    return gu.transitivity_avglocal_undirected(mode="zero")


def triangle_count(g):
    """Exact triangle count. T = (A^2 .* A).sum() / 6  for symmetric 0/1 A."""
    A = _adjacency(g)
    return int(round((A @ A).multiply(A).sum() / 6.0))


def square_count(g):
    """Exact 4-cycle count. C4 = (1/4) * sum_{i!=j} C(c_ij, 2), c = common neighbours."""
    A2 = (_adjacency(g) @ _adjacency(g)).tolil()
    A2.setdiag(0)
    c = A2.tocsr().data
    return int(round(np.sum(c * (c - 1) / 2.0) / 4.0))

def orbit_dist(g, graphlet_size=4):
    return orbit_count.node_orbit_counts(g, graphlet_size=graphlet_size)


def characteristic_path_length(g):
    """Average shortest path length over reachable pairs (undirected). Small scale only."""
    return _undirected(g).average_path_length(directed=False, unconn=True)
    

def leiden(g, seed=0):
    """Leiden partition (modularity objective) on the undirected graph."""
    random.seed(seed)
    gu = _undirected(g)
    return gu.community_leiden(objective_function="modularity")


def modularity(part):
    return part.modularity


def inter_intra_ratio(part):
    """#edges between communities / #edges within communities."""
    crossing = np.asarray(part.crossing())
    inter = int(crossing.sum())
    intra = crossing.size - inter
    return inter / intra if intra else float("inf")


def degree_sequence(g, mode="all"):
    """mode in {'in', 'out', 'all'}. Directed graphs honour the mode; undirected ignore it."""
    if g.is_directed():
        fn = {"in": g.indegree, "out": g.outdegree, "all": g.degree}[mode]
        return np.array(fn())
    return np.array(g.degree())


def community_size_sequence(part):
    return np.array(part.sizes())


def laplacian_eigenvalues(g):
    """Full Laplacian spectrum (dense eigvalsh). Small scale only."""
    A = _adjacency(g).toarray()
    L = np.diag(A.sum(1)) - A
    return np.linalg.eigvalsh(L)


def w1_distance(sample_gen, sample_ref):
    """1-Wasserstein distance between two 1D samples. Best = 0."""
    return float(wasserstein_distance(np.asarray(sample_gen, dtype=float), np.asarray(sample_ref, dtype=float)))