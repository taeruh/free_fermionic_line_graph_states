import numpy as np
from line_graph_solution.mapping import BaseMapping
from hamiltonian import Hamiltonian
from paulis import Pauli
from numpy.typing import NDArray

from typing import Tuple


def get_exponent_sequence(
    angles: list[Tuple[int, int, np.float64]],
    mapping: BaseMapping,
    hamiltonian: Hamiltonian,
) -> list[Tuple[np.float64, Pauli]]:
    """we return a list of (theta, pauli) and the actual exponent is \"2 i theta pauli\" """
    ret = []
    for u, v, angle in angles:
        p = mapping.psi_to_pauli(u - 1, v - 1, hamiltonian)
        # print(u - 1, v - 1)
        # print(p.to_pauli_string(), p)
        # print(p.to_matrix())
        # print()
        ret.append((angle / 4.0, p))
    return ret


def get_matchgates(exponents: list[Tuple[np.float64, Pauli]]) -> NDArray[np.complex128]:
    ret = []
    dim = 2 ** exponents[0][1].n
    for angle, pauli in exponents:
        # print(angle, pauli.to_pauli_string())
        ret.append(
            np.cos(2 * angle) * np.identity(dim, dtype=np.complex128)
            + 1j * np.sin(2 * angle) * pauli.to_matrix()
        )
    return ret  # pyright: ignore
