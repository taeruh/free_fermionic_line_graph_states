import numpy as np
from numpy.typing import NDArray

from linalg.f2 import matrix
from linalg.f2.symplectic import SymplecticVector
from linalg.f2.matrix import Vector


class Pauli(SymplecticVector):
    """
    the actual phase is i^phase and we encode ZX as (z=1, x=1, phase=0), i.e., it is Y =
    -iZX <-> (1, 1, 3);
    """

    def __init__(self, n: int, z: NDArray[np.bool], x: NDArray[np.bool], phase: int):
        assert len(z) == n
        assert len(x) == n
        assert phase in (0, 1, 2, 3)
        self.phase = phase
        super().__init__(np.concatenate([z, x]))

    def clone(self) -> "Pauli":
        return Pauli(self.n, self.u.copy(), self.l.copy(), self.phase)

    def __repr__(self):
        return f"({self.u.astype(np.int_)},{self.l.astype(np.int_)}; {self.n}, {self.phase})"

    @classmethod
    def identity(cls, n: int) -> "Pauli":
        return cls(n, np.zeros(n, dtype=np.bool), np.zeros(n, dtype=np.bool), 0)

    @classmethod
    def from_symplectic_vector(cls, sv: SymplecticVector, sign: int) -> "Pauli":
        return Pauli(sv.n, sv.u, sv.l, sign)

    def get_hermitian_phase(self) -> int:
        """
        the phase when writing the pauli as X/Y/Z string
        """
        phase = self.phase
        for i in range(self.n):
            if self.u[i] and self.l[i]:
                # multiply by 1 = i * -i and take i into the phase and make ZX to Y
                phase = (phase + 1) % 4
        return phase

    def to_string(self, flip: bool = False, with_phase: bool = True) -> str:
        """in the string, we order from right to left, i.e., p_n-1, ... p_0 (as in a
        bitvector)"""
        pauli_str = ""
        if flip:
            iter = range(self.n)
        else:
            iter = range(self.n - 1, -1, -1)
        for i in iter:
            if self.u[i] and self.l[i]:
                pauli_str += "Y"
            elif self.u[i]:
                pauli_str += "Z"
            elif self.l[i]:
                pauli_str += "X"
            else:
                pauli_str += "I"
        if with_phase:
            phase = self.get_hermitian_phase()
            pauli_str += ", "
            if phase == 0:
                pauli_str += "+1"
            elif phase == 1:
                pauli_str += "+i"
            elif phase == 2:
                pauli_str += "-1"
            else:
                pauli_str += "-i"
        return pauli_str

    def to_matrix(self) -> NDArray[np.complex128]:
        """
        Return the matrix representation of the Pauli operator. The matrix indices are
        ordered as bitvectors, i.e., 0 = 000, 1 = 001, 2 = 010, 3 = 011, etc., where the
        paulis are associated to qubits from right to left (cf. to_string).
        """
        pauli_matrices = {
            (0, 0): np.array([[1, 0], [0, 1]], dtype=complex),
            (1, 0): np.array([[1, 0], [0, -1]], dtype=complex),
            (0, 1): np.array([[0, 1], [1, 0]], dtype=complex),
            (1, 1): np.array([[0, 1], [-1, 0]], dtype=complex),
        }
        result = np.array([[1]], dtype=complex)
        for i in range(self.n):
            z = int(self.u[i])
            x = int(self.l[i])
            result = np.kron(pauli_matrices[(z, x)], result)
        return np.complex128((1j) ** self.phase) * result

    def is_proportional_to(self, other: "Pauli") -> bool:
        return np.array_equal(self.data, other.data)

    def is_proportional_to_identity(self) -> bool:
        return np.array_equal(self.data, np.zeros(2 * self.n, dtype=np.bool))

    def is_identity(self) -> bool:
        return self.phase == 0 and self.is_proportional_to_identity()

    def multiply_as_paulis(self, other: "Pauli") -> "Pauli":
        assert self.n == other.n
        new_z = self.u ^ other.u
        new_x = self.l ^ other.l
        # determine the phase
        new_phase = (self.phase + other.phase) % 4
        for i in range(self.n):
            if self.l[i] and other.u[i]:
                new_phase = (new_phase + 2) % 4
        return Pauli(self.n, new_z, new_x, new_phase)


def decompose_into_independent_paulis(
    independent_paulis, pauli
) -> tuple[int, list[int]] | None:
    decomposition = matrix.linear_decomposition(independent_paulis, pauli)
    if decomposition is None:
        return None
    # otherwise it would be an identity, which we probably should have filtered out before
    # calling this function here
    assert len(decomposition) > 0
    pauli_up_to_sign = independent_paulis[decomposition[0]]
    for i in decomposition[1:]:
        pauli_up_to_sign = pauli_up_to_sign.multiply_as_paulis(independent_paulis[i])
    assert pauli_up_to_sign.is_proportional_to(pauli)
    identity_with_phase = pauli.multiply_as_paulis(pauli_up_to_sign)
    return identity_with_phase.phase, decomposition
