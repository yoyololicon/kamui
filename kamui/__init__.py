import numpy as np
from typing import Tuple, Optional, Iterable, Union

from .core import *
from .utils import *


def wrap_difference(x: np.ndarray, wrapped_interval: float = 2 * np.pi) -> np.ndarray:
    """
    Wrap a 1D array.
    Args:
        x: (N,) array
        wrapped_interval: interval of the wrapped axis
    Returns:
        (N,) array
    """
    return np.mod(x + wrapped_interval / 2, wrapped_interval) - wrapped_interval / 2


def unwrap(
    x: np.ndarray,
    wrapped_interval: float = 2 * np.pi,
    start_pixel: Optional[Union[Tuple[int, int], Tuple[int, int, int]]] = None,
) -> np.ndarray:
    """
    Unwrap a 2D or 3D array.
    Args:
        x: (N, M) or (N, M, L) array
        wrapped_interval: interval of the wrapped axis
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
    diff = wrap_difference(psi[edges[:, 1]] - psi[edges[:, 0]], wrapped_interval)
    k = calculate_k(edges, simplices, diff / wrapped_interval)
    correct_diff = diff + k * wrapped_interval
    start_i = start_pixel[0] * x.shape[1] + start_pixel[1]
    result = (
        integrate(
            np.concatenate((edges, edges[:, ::-1]), axis=0),
            np.concatenate((correct_diff, -correct_diff), axis=0),
            start_i=start_i,
        )
        + psi[start_i]
    )
    return result.reshape(x.shape)
