import numpy as np
from typing import Tuple, Optional, Iterable, Union

__all__ = ["get_2d_edges_and_simplices", "get_3d_edges_and_simplices"]


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
