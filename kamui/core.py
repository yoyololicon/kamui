from scipy.sparse import csgraph as csg
import scipy.sparse as sp
from scipy.optimize import linprog
import numpy as np
from typing import Optional, Iterable

__all__ = ["integrate", "calculate_k", "calculate_m"]


def integrate(edges: np.ndarray, weights: np.ndarray, start_i: int = 0):
    """
    Args:
        edges: (N, 2) array of edges
        weights: (N,) array of weights
        start_i: starting index
    Returns:
        (N,) array of integrated weights
    """
    G = sp.csr_matrix((weights, (edges[:, 0], edges[:, 1])))
    N = max(G.shape)
    result = np.zeros(N, dtype=weights.dtype)

    nodes = csg.depth_first_order(G, start_i, directed=True, return_predecessors=False)

    pairs = np.stack([nodes[:-1], nodes[1:]], axis=1)
    for u, v in pairs:
        result[v] = result[u] + G[u, v]

    return result


def calculate_k(
    edges: np.ndarray,
    simplices: Iterable[Iterable[int]],
    differences: np.ndarray,
    weights: Optional[np.ndarray] = None,
    adaptive_weighting: bool = True,
) -> Optional[np.ndarray]:
    """
    Args:
        edges: (M, 2) array of edges
        simplices: (N,) iterable of simplices
        differences: (M,) array of differences, could be float or int
        weights: (M,) array of weights
        adaptive_weighting: whether to use adaptive weighting
    """

    M, N = edges.shape[0], len(simplices)

    edge_dict = {tuple(x): i for i, x in enumerate(edges)}
    if len(edge_dict) != M:
        raise ValueError("edges must be unique")
    rows = []
    cols = []
    vals = []
    for i, simplex in enumerate(simplices):
        u = simplex[-1]
        for v in simplex:
            key = (u, v)
            rows.append(i)
            if key in edge_dict:
                cols.append(edge_dict[key])
                vals.append(1)
            else:
                try:
                    cols.append(edge_dict[(v, u)])
                except KeyError:
                    raise ValueError("simplices contain invalid edges")
                vals.append(-1)
            u = v

    rows = np.array(rows)
    cols = np.array(cols)
    vals = np.array(vals)

    V = sp.csr_matrix((vals, (rows, cols)), shape=(N, M))
    y = V @ differences
    y = np.round(y).astype(np.int64)

    A_eq = sp.csr_matrix(
        (
            np.concatenate((vals, -vals)),
            (np.tile(rows, 2), np.concatenate((cols, cols + M))),
        ),
        shape=(N, M * 2),
    )
    b_eq = -y

    if weights is None:
        if adaptive_weighting:
            nonzero_simplices = np.minimum(np.abs(b_eq), 1)
            W = np.abs(A_eq)
            num_nonzero_simplices = nonzero_simplices @ W
            num_simplices = W.sum(0).A1
            c = num_simplices - num_nonzero_simplices
        else:
            c = np.ones((M * 2,), dtype=np.int64)
    else:
        c = np.tile(c, 2)

    res = linprog(c, A_eq=A_eq, b_eq=b_eq, integrality=1)
    k = res.x[:M] - res.x[M:]
    k = k.astype(np.int64)
    cost = np.abs(V @ k + y).sum()
    if cost > 0:
        print(f"Warning: no solution found.")
        return None
    return k


def calculate_m(
    edges: np.ndarray,
    differences: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """
    Args:
        edges: (M, 2) array of edges
        differences: (M,) array of differences, must be int
        weights: (M,) array of weights
    """
    assert differences.dtype == np.int64, "differences must be int"
    M = edges.shape[0]
    N = np.max(edges) + 1

    vals = np.concatenate(
        (np.ones((M,), dtype=np.int64), -np.ones((M,), dtype=np.int64))
    )
    rows = np.tile(np.arange(M), 2)
    cols = np.concatenate((edges[:, 0], edges[:, 1]))

    A_eq = sp.csr_matrix(
        (
            np.concatenate((vals, -vals, np.ones(M), -np.ones(M))).astype(np.int64),
            (
                np.tile(rows, 3),
                np.concatenate((cols, cols + N, np.arange(2 * M) + 2 * N)),
            ),
        ),
        shape=(M, 2 * N + 2 * M),
    )
    if weights is None:
        weights = np.ones((M,), dtype=np.int64)

    c = np.concatenate((np.zeros(2 * N, dtype=np.int64), weights, weights))

    b_eq = differences

    res = linprog(c, A_eq=A_eq, b_eq=b_eq, integrality=1)
    if res.x is None:
        return None
    m = res.x[:N] - res.x[N : 2 * N]
    m = m.astype(np.int64)
    return m
