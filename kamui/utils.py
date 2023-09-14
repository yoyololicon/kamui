import numpy as np
from typing import Tuple, Optional, Iterable

__all__ = ["get_2d_edges_and_simplices", "get_3d_edges_and_simplices"]


def get_2d_edges_and_simplices(
    shape: Tuple[int, int]
) -> Tuple[np.ndarray, Iterable[Iterable[int]]]:
    nodes = np.arange(np.prod(shape)).reshape(shape)
    edges = np.concatenate(
        (
            np.stack([nodes[:, :-1].ravel(), nodes[:, 1:].ravel()], axis=1),
            np.stack([nodes[:-1, :].ravel(), nodes[1:, :].ravel()], axis=1),
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
    return edges, simplices


def get_3d_edges_and_simplices(
    shape: Tuple[int, int, int]
) -> Tuple[np.ndarray, Iterable[Iterable[int]]]:
    nodes = np.arange(np.prod(shape)).reshape(shape)
    edges = np.concatenate(
        (
            np.stack([nodes[:, :-1, :].ravel(), nodes[:, 1:, :].ravel()], axis=1),
            np.stack([nodes[:-1, :, :].ravel(), nodes[1:, :, :].ravel()], axis=1),
            np.stack([nodes[:, :, :-1].ravel(), nodes[:, :, 1:].ravel()], axis=1),
        ),
        axis=0,
    )
    simplices = np.concatenate(
        (
            np.stack(
                (
                    nodes[:-1, :-1, :-1].ravel(),
                    nodes[1:, :-1, :-1].ravel(),
                    nodes[1:, 1:, :-1].ravel(),
                    nodes[:-1, 1:, :-1].ravel(),
                ),
                axis=1,
            ),
            np.stack(
                (
                    nodes[:-1, :-1, :-1].ravel(),
                    nodes[:-1, 1:, :-1].ravel(),
                    nodes[:-1, 1:, 1:].ravel(),
                    nodes[:-1, :-1, 1:].ravel(),
                ),
                axis=1,
            ),
            np.stack(
                (
                    nodes[:-1, :-1, :-1].ravel(),
                    nodes[:-1, :-1, 1:].ravel(),
                    nodes[1:, :-1, 1:].ravel(),
                    nodes[1:, :-1, :-1].ravel(),
                ),
                axis=1,
            ),
        ),
        axis=0,
    ).tolist()
    return edges, simplices
