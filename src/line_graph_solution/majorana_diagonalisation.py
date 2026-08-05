import numpy as np
from line_graph_solution.mapping import BaseMapping
from numpy.typing import NDArray
import scipy.linalg as linalg
from helper import numpy as np_helper
from enum import Enum
from typing import Tuple, Union


def get_h(mapping: BaseMapping, vertex_weights) -> NDArray[np.float64]:
    m = mapping.num_modes
    h = np.zeros((m, m), dtype=np.float64)
    for v, (s, e) in mapping.phi.items():
        w = ((-1) ** s) * vertex_weights[v] / 2
        h[e[0], e[1]] += w
        h[e[1], e[0]] -= w
    return h


def skew_diagonalise(
    h: NDArray[np.float64],
    block_sizes: list[int],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """return (Lambda, K) such that h = K lambda K^dagger"""
    lm, km = linalg.schur(h, output="real")  # pyright: ignore

    lblocks = np_helper.view_blocks(lm, block_sizes)
    kblocks = np_helper.view_blocks(km, block_sizes)
    #
    for lb, kb in zip(lblocks, kblocks):
        det = np.linalg.det(kb)
        if det < 0:
            d = np.identity(kb.shape[0], dtype=np.float64)
            d[0, 0] = -1
            kb[:] = kb @ d
            lb[:] = d @ lb @ d
    # m, n = lm.shape
    # print(sum(np.abs(h - km @ lm @ km.T).flatten()))
    # for i in range(m):
    #     for j in range(n):
    #         if np.abs(lm[i, j]) < 1e-10:
    #             lm[i, j] = 0.0
    # print(sum(np.abs(h - km @ lm @ km.T).flatten()))
    # print(h - km @ lm @ km.T)
    # print(h)
    assert np.allclose(h, km @ lm @ km.T)
    return lm, km  # pyright: ignore
