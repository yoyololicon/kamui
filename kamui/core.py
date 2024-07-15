from scipy.sparse import csgraph as csg
import scipy.sparse as sp
from scipy.optimize import linprog
import numpy as np
from typing import Optional, Iterable


try:
    import maxflow
except ImportError:
    print("PyMaxflow not found, some functions will not be available.")
__all__ = ["integrate", "calculate_k", "calculate_m", "puma"]


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
        c = np.tile(weights, 2)
    res = linprog(c, A_eq=A_eq, b_eq=b_eq, integrality=1)
    if res.x is None:
        return None
    k = res.x[:M] - res.x[M:]
    k = k.astype(np.int64)
    return k


def calculate_m(
    edges: np.ndarray,
    differences: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """
    Args:
        edges: (M, 2) array of edges
        differences: (M,) quantised array of differences, must be int
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
            np.concatenate((vals, np.ones(M), -np.ones(M))).astype(np.int64),
            (
                np.tile(rows, 2),
                np.concatenate((cols, np.arange(2 * M) + N)),
            ),
        ),
        shape=(M, N + 2 * M),
    )
    if weights is None:
        weights = np.ones((M,), dtype=np.int64)
    c = np.concatenate((np.zeros(N, dtype=np.int64), weights, weights))

    b_eq = differences

    res = linprog(c, A_eq=A_eq, b_eq=b_eq, integrality=1)
    if res.x is None:
        return None
    m = res.x[:N]
    return m.astype(np.int64)


def puma(psi: np.ndarray, edges: np.ndarray, max_jump: int = 1, p: float = 1):
    """
    Args:
        psi: (N,) array
        edges: (M, 2) array of edges
        max_jump: maximum jump step
        p: p-norm
    Returns:
        (N,) array
    """
    if max_jump > 1:
        jump_steps = list(range(1, max_jump + 1)) * 2
    else:
        jump_steps = [max_jump]
    total_nodes = psi.size

    def V(x):
        return np.abs(x) ** p

    K = np.zeros_like(psi)

    def cal_Ek(K, psi, i, j):
        return np.sum(V(K[j] - K[i] - psi[i] + psi[j]))

    prev_Ek = cal_Ek(K, psi, edges[:, 0], edges[:, 1])

    for step in jump_steps:
        while 1:
            G = maxflow.Graph[float]()
            G.add_nodes(total_nodes)

            i, j = edges[:, 0], edges[:, 1]
            psi_diff = psi[i] - psi[j]
            a = (K[j] - K[i]) - psi_diff
            e00 = e11 = V(a)
            e01 = V(a - step)
            e10 = V(a + step)
            weight = np.maximum(0, e10 + e01 - e00 - e11)

            G.add_edges(edges[:, 0], edges[:, 1], weight, np.zeros_like(weight))

            a = e10 - e00
            flip_mask = a < 0
            tmp_st_weight = np.zeros((2, total_nodes))
            flip_index = np.stack(
                (flip_mask.astype(int), 1 - flip_mask.astype(int)), axis=1
            )
            positive_a = np.where(flip_mask, -a, a)
            np.add.at(
                tmp_st_weight, (flip_index.ravel(), edges.ravel()), positive_a.repeat(2)
            )

            for i in range(total_nodes):
                G.add_tedge(i, tmp_st_weight[0, i], tmp_st_weight[1, i])
            G.maxflow()

            partition = G.get_grid_segments(np.arange(total_nodes))
            K[~partition] += step

            energy = cal_Ek(K, psi, edges[:, 0], edges[:, 1])

            if energy < prev_Ek:
                prev_Ek = energy
            else:
                K[~partition] -= step
                break
    return K
