from linalg.f2 import matrix, symplectic
import paulis
from paulis import Pauli
from qiskit.quantum_info import Clifford
from numpy.typing import NDArray
from typing import Tuple
import numpy as np


def get_clifford_from_paulis(
    ops: list[Pauli],
    do_check: bool = True,
) -> Tuple[Clifford, list[Pauli], NDArray[np.complex128]]:
    """
    Given some stabiliser generators, return the corresponding Clifford that conjugates
    the Z stabilisers to these generators, and additional Pauli conjugations to fix some
    signs. The Unitary is given by U = P[1] ... P[k] C, and we conjugate as
    ops[i] = U Z_i U^dagger (The order of the Paulis does not matter as they commute,
    however, they have to applied after the Clifford as they are define to swap the sign
    on ops[i] but not Z_i). We also return U (as last argument) as matrix for convenience.
    """
    assert symplectic.stabilisers_all_commuting(ops)  # pyright: ignore
    full: list[Pauli] = []
    for p in symplectic.extend_to_full_isotropic_set(ops):  # pyright: ignore
        if not isinstance(p, Pauli):
            full.append(Pauli.from_symplectic_vector(p, 1))
        else:
            full.append(p)
    assert symplectic.stabilisers_all_commuting(full)  # pyright: ignore
    assert len(full) == full[0].n
    # they might not be hermitian, but we don't care about that (they are still unitaries)
    pairs = [
        Pauli.from_symplectic_vector(p, 0)
        for p in symplectic.get_full_hyperbolic_partners(full)  # pyright: ignore
    ]
    pauli_dict = {
        "stabilizer": [p.to_string(with_phase=False) for p in full],
        "destabilizer": [p.to_string(with_phase=False) for p in pairs],
    }
    cliff = Clifford.from_dict(pauli_dict)
    cliffm: NDArray = cliff.to_matrix()  # pyright: ignore
    additional_sign_corrections = []
    for i, pauli in enumerate(ops):
        if pauli.get_hermitian_phase() == 2:
            additional_correction = pairs[i]
            additional_sign_corrections.append(additional_correction)

    full_cliff = cliffm
    for p in additional_sign_corrections:
        full_cliff = p.to_matrix() @ full_cliff

    # complete check:
    if do_check:
        for i, p in enumerate(ops):
            z = Pauli.identity(p.n)
            a = p.to_matrix()
            z.u[i] = True
            zm = z.to_matrix()
            assert np.allclose(full_cliff @ zm @ full_cliff.conj().T, p.to_matrix())

    return cliff, additional_sign_corrections, full_cliff
