import numpy as np
import numpy.typing as npt
from typing import Tuple, Iterable, Union

__all__ = [
    "get_2d_edges_and_simplices",
    "get_3d_edges_and_simplices",
    "prepare_weights",
]


def get_2d_edges_and_simplices(
    shape: Tuple[int, int], cyclical_axis: Union[int, Tuple[int, int]] = ()
) -> Tuple[np.ndarray, Iterable[Iterable[int]]]:
    """
    Compute the edges and simplices for a 2D grid.

    Parameters
    ----------
    shape : Tuple[int, int]
        The shape of the grid.
    cyclical_axis : Union[int, Tuple[int, int]], optional
        The axis/axes that should be treated as cyclical. Defaults to ().

    Returns
    -------
    Tuple[np.ndarray, Iterable[Iterable[int]]]
        A tuple containing the edges and simplices of the grid.
    """
    nodes = np.arange(np.prod(shape)).reshape(shape)
    if type(cyclical_axis) is int:
        cyclical_axis = (cyclical_axis,)
    # if the axis length <= 2, then the axis is already cyclical

    cyclical_axis = tuple(filter(lambda ax: shape[ax] > 2, cyclical_axis))

    edges = np.concatenate(
        (
            np.stack([nodes[:, :-1].ravel(), nodes[:, 1:].ravel()], axis=1),
            np.stack([nodes[:-1, :].ravel(), nodes[1:, :].ravel()], axis=1),
        )
        + tuple(
            np.stack(
                [
                    np.take(nodes, [0], axis=ax).ravel(),
                    np.take(nodes, [-1], axis=ax).ravel(),
                ],
                axis=1,
            )
            for ax in cyclical_axis
        ),
        axis=0,
    )
    simplices = np.stack(
        (
            nodes[:-1, :-1].ravel(),
            nodes[1:, :-1].ravel(),
            nodes[1:, 1:].ravel(),
            nodes[:-1, 1:].ravel(),
        ),
        axis=1,
    ).tolist()
    if len(cyclical_axis) > 0:
        pairs = [
            (
                np.squeeze(np.take(nodes, [0], axis=ax), axis=ax),
                np.squeeze(np.take(nodes, [-1], axis=ax), axis=ax),
            )
            for ax in cyclical_axis
        ]
        simplices += np.concatenate(
            tuple(
                np.stack(
                    (
                        x[:-1],
                        y[:-1],
                        y[1:],
                        x[1:],
                    ),
                    axis=1,
                )
                for x, y in pairs
            ),
            axis=0,
        ).tolist()
    return edges, simplices


def get_3d_edges_and_simplices(
    shape: Tuple[int, int, int], cyclical_axis: Union[int, Tuple[int, int]] = ()
) -> Tuple[np.ndarray, Iterable[Iterable[int]]]:
    """
    Compute the edges and simplices for a 3D grid.

    Parameters
    ----------
    shape : Tuple[int, int, int]
        The shape of the grid.
    cyclical_axis : Union[int, Tuple[int, int]], optional
        The axis/axes that should be treated as cyclical. Defaults to ().

    Returns
    -------
    Tuple[np.ndarray, Iterable[Iterable[int]]]
        A tuple containing the edges and simplices of the grid.
    """
    nodes = np.arange(np.prod(shape)).reshape(shape)
    if type(cyclical_axis) is int:
        cyclical_axis = (cyclical_axis,)
    cyclical_axis = tuple(filter(lambda ax: shape[ax] > 2, cyclical_axis))

    edges = np.concatenate(
        (
            np.stack([nodes[:, :-1, :].ravel(), nodes[:, 1:, :].ravel()], axis=1),
            np.stack([nodes[:-1, :, :].ravel(), nodes[1:, :, :].ravel()], axis=1),
            np.stack([nodes[:, :, :-1].ravel(), nodes[:, :, 1:].ravel()], axis=1),
        )
        + tuple(
            np.stack(
                [
                    np.take(nodes, [0], axis=ax).ravel(),
                    np.take(nodes, [-1], axis=ax).ravel(),
                ],
                axis=1,
            )
            for ax in cyclical_axis
        ),
        axis=0,
    )
    simplices = np.concatenate(
        (
            np.stack(
                (
                    nodes[:-1, :-1, :].ravel(),
                    nodes[1:, :-1, :].ravel(),
                    nodes[1:, 1:, :].ravel(),
                    nodes[:-1, 1:, :].ravel(),
                ),
                axis=1,
            ),
            np.stack(
                (
                    nodes[:, :-1, :-1].ravel(),
                    nodes[:, 1:, :-1].ravel(),
                    nodes[:, 1:, 1:].ravel(),
                    nodes[:, :-1, 1:].ravel(),
                ),
                axis=1,
            ),
            np.stack(
                (
                    nodes[:-1, :, :-1].ravel(),
                    nodes[:-1, :, 1:].ravel(),
                    nodes[1:, :, 1:].ravel(),
                    nodes[1:, :, :-1].ravel(),
                ),
                axis=1,
            ),
        ),
        axis=0,
    ).tolist()

    if len(cyclical_axis) > 0:
        simplices += np.concatenate(
            sum(
                (
                    (
                        np.stack(
                            (
                                x[1:, :].ravel(),
                                y[1:, :].ravel(),
                                y[:-1, :].ravel(),
                                x[:-1, :].ravel(),
                            ),
                            axis=1,
                        ),
                        np.stack(
                            (
                                x[:, 1:].ravel(),
                                y[:, 1:].ravel(),
                                y[:, :-1].ravel(),
                                x[:, :-1].ravel(),
                            ),
                            axis=1,
                        ),
                    )
                    for x, y in [
                        (
                            np.squeeze(np.take(nodes, [0], axis=ax), axis=ax),
                            np.squeeze(np.take(nodes, [-1], axis=ax), axis=ax),
                        )
                        for ax in cyclical_axis
                    ]
                ),
                (),
            ),
            axis=0,
        ).tolist()
    return edges, simplices


def prepare_weights(
    weights: npt.NDArray,
    edges: npt.NDArray[np.int_],
    smoothing: float = 0.1,
    merging_method: str = "mean",
) -> npt.NDArray[np.float_]:
    """Prepare weights for `calculate_m` and `calculate_k` functions.

    Assume the weights are the same shape as the phases to be unwrapped.

    Scale the weights from 0 to 1. Pick the weights corresponding to the phase pairs connected by the edges.
    Compute the mean/max/min (depending on the `merging_method`) of each of those pairs to give a weight for each edge.

    Args:
        weights         :   Array of weights of shape corresponding to the original phases array shape.
        edges           :   Edges connecting the phases. Shape: (M, 2), where M is the number of edges.
        smoothing       :   A positive value in range [0, 1). This is the minimal value of the rescaled weights
                            where they are defined. If smoothing > 0, the value of 0 is reserved for places where
                            the weights are originally NaN. If smoothing == 0, 0 will be used for both NaN weights
                            and smallest non-NaN ones.
        merging_method  :   Way of combining two phase weights into a single edge weight.

    Returns:
        Array of weights for the edges, shape: (M,). Rescaled to [0, 1].
    """

    if not 0 <= smoothing < 1:
        raise ValueError(
            "`smoothing` should be a value between 0 (inclusive) and 1 (non inclusive); got "
            + str(smoothing)
        )
    # scale the weights from 0 to 1

    weights = weights - np.nanmin(weights)
    current_max = np.nanmax(weights)
    if not current_max:
        # current maximum is 0, which means all weights originally had the same value, now 0; replace everything with 1

        weights += 1
    else:
        weights /= current_max
        weights *= 1 - smoothing
        weights += smoothing
    # pick the weights corresponding to the phases connected by the edges
    # and use `merging_method` to get one weight for each edge

    allowed_merging_methods = ["min", "max", "mean"]
    if merging_method not in allowed_merging_methods:
        raise ValueError(
            "`merging_method` should be one of: "
            + ", ".join(merging_method)
            + "; got "
            + str(merging_method)
        )
    weights_for_edges = getattr(np, merging_method)(weights.ravel()[edges], axis=1)

    # make sure there are no NaNs in the weights; replace any with 0s

    weights_for_edges[np.isnan(weights_for_edges)] = 0

    return weights_for_edges
