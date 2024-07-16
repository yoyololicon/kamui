import numpy as np
from typing import Tuple, Optional, Iterable, Union, Any

from .core import *
from .utils import *

__all__ = [
    "unwrap_dimensional",
    "unwrap_arbitrary",
]


def wrap_difference(x: np.ndarray, period: float = 2 * np.pi) -> np.ndarray:
    """
    Wrap a 1D array.
    Args:
        x: (N,) array
        period: interval of the wrapped axis
    Returns:
        (N,) array
    """
    return np.mod(x + period / 2, period) - period / 2


def unwrap_dimensional(
    x: np.ndarray,
    start_pixel: Optional[Union[Tuple[int, int], Tuple[int, int, int]]] = None,
    use_edgelist: bool = False,
    cyclical_axis: Union[int, Tuple[int, int]] = (),
    merging_method: str = "mean",
    weights: Optional[np.ndarray] = None,
    **kwargs: Any,
) -> Optional[np.ndarray]:
    """
    Unwrap the phase of a 2-D or 3-D array.

    Parameters
    ----------
    x : 2-D or 3-D np.ndarray
        The phase to be unwrapped.
    start_pixel : (2,) or (3,) tuple
        the reference pixel to start unwrapping.
        Default to (0, 0) for 2-D data and (0, 0, 0) for 3-D data.
    use_edgelist : bool
        Whether to use the edgelist method.
        Default to False.
    cyclical_axis : int or (int, int)
        The axis that is cyclical.
        Default to ().
    weights : Weights defining the 'goodness' of value at each vertex. Shape must match the shape of x.
    merging_method : Way of combining two phase weights into a single edge weight.
    kwargs : dict
        Other arguments passed to `kamui.unwrap_arbitrary`.

    Returns
    -------
    np.ndarray
        The unwrapped phase of the same shape as x.
    """
    if start_pixel is None:
        start_pixel = (0,) * x.ndim
    assert x.ndim == len(start_pixel), "start_pixel must have the same dimension as x"

    start_i = 0
    for i, s in enumerate(start_pixel):
        start_i *= x.shape[i]
        start_i += s
    if x.ndim == 2:
        edges, simplices = get_2d_edges_and_simplices(
            x.shape, cyclical_axis=cyclical_axis
        )
    elif x.ndim == 3:
        edges, simplices = get_3d_edges_and_simplices(
            x.shape, cyclical_axis=cyclical_axis
        )
    else:
        raise ValueError("x must be 2D or 3D")
    psi = x.ravel()

    if weights is not None:
        # convert per-vertex weights to per-edge weights

        weights = prepare_weights(weights, edges=edges, merging_method=merging_method)
    result = unwrap_arbitrary(
        psi,
        edges,
        None if use_edgelist else simplices,
        start_i=start_i,
        weights=weights,
        **kwargs,
    )
    if result is None:
        return None
    return result.reshape(x.shape)


def unwrap_arbitrary(
    psi: np.ndarray,
    edges: np.ndarray,
    simplices: Iterable[Iterable[int]] = None,
    method: str = "ilp",
    period: float = 2 * np.pi,
    start_i: int = 0,
    **kwargs,
) -> Optional[np.ndarray]:
    """
    Unwrap the phase of arbitrary data.

    Parameters
    ----------
    psi : 1D np.ndarray of shape (P,)
        The phase (vertices) to be unwrapped.
    edges : 2-D np.ndarray of shape (M, 2)
        The edges of the graph.
    simplices : Iterable[Iterable[int]] of length (N,)
        Each element is a list of vertices that form a simplex (a.k.a elementary cycle).
        The connections should be consistent with the edges.
        This is also used to compute automatic weights for each edge.
        If not provided and method is "ilp", an edgelist-based ILP solver will be used without weighting.
    method : str
        The method to be used. Valid options are "ilp" and "gc", where "gc" correponds to PUMA.
        Default to "ilp".
    period : float
        The period of the phase.
        Default to 2 * np.pi.
    start_i : int
        The index of the reference vertex to start unwrapping.
        Default to 0.
    kwargs : dict
        Other arguments passed to the solver.

    Returns
    -------
    np.ndarray
        The unwrapped phase of the same shape as psi.
    """
    if method == "gc":
        m = puma(psi / period, edges, **kwargs)
        m -= m[start_i]
        result = m * period + psi
    elif method == "ilp":
        if simplices is None:
            m = calculate_m(
                edges,
                np.round((psi[edges[:, 1]] - psi[edges[:, 0]]) / period).astype(
                    np.int64
                ),
                **kwargs,
            )
            if m is None:
                return None
            m -= m[start_i]
            result = m * period + psi
        else:
            diff = wrap_difference(psi[edges[:, 1]] - psi[edges[:, 0]], period)
            k = calculate_k(edges, simplices, diff / period, **kwargs)
            correct_diff = diff + k * period

            result = (
                integrate(
                    np.concatenate((edges, np.flip(edges, 1)), axis=0),
                    np.concatenate((correct_diff, -correct_diff), axis=0),
                    start_i=start_i,
                )
                + psi[start_i]
            )
    else:
        raise ValueError("method must be 'gc' or 'ilp'")
    return result
