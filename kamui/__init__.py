import numpy as np
from typing import Tuple, Optional, Iterable, Union

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
    **kwargs,
) -> np.ndarray:
    """
    Unwrap a 2D or 3D array.
    Args:
        x: (N, M) or (N, M, L) array
        period: interval of the wrapped axis
    Returns:
        (N, M) or (N, M, L) array
    """
    if start_pixel is None:
        start_pixel = (0,) * x.ndim
    assert x.ndim == len(start_pixel), "start_pixel must have the same dimension as x"

    start_i = 0
    for i, s in enumerate(start_pixel):
        start_i *= x.shape[i]
        start_i += s

    if x.ndim == 2:
        edges, simplices = get_2d_edges_and_simplices(x.shape)
    elif x.ndim == 3:
        edges, simplices = get_3d_edges_and_simplices(x.shape)
    else:
        raise ValueError("x must be 2D or 3D")
    psi = x.ravel()

    result = unwrap_arbitrary(psi, edges, simplices, start_i=start_i, **kwargs)
    if result is None:
        return None

    return result.reshape(x.shape)


def unwrap_arbitrary(
    psi: np.ndarray,
    edges: np.ndarray,
    simplices: Iterable[Iterable[int]] = None,
    period: float = 2 * np.pi,
    start_i: int = 0,
    **kwargs,
) -> Optional[np.ndarray]:
    """
    Unwrap an arbitrary array.
    Args:
        psi: (N,) array
        edges: (M, 2) array of edges
        simplices: (N,) iterable of simplices
        period: interval of the wrapped axis
        start_i: starting index
    Returns:
        (N,) array
    """
    if simplices is None:
        m = calculate_m(
            edges,
            np.round((psi[edges[:, 1]] - psi[edges[:, 0]]) / period).astype(np.int64),
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
    return result
