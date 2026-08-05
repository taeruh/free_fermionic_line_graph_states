import numpy as np
from numpy.typing import NDArray


def view_blocks(
    matrix: NDArray[np.float64], block_sizes: list[int]
) -> list[NDArray[np.float64]]:
    blocks = []
    sizes = [0] + block_sizes
    start = 0
    end = 0
    for i in range(len(sizes) - 1):
        start += sizes[i]
        end += sizes[i + 1]
        block = matrix[start:end, start:end]
        blocks.append(block)
    return blocks
