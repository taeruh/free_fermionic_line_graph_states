import numpy as np
from sage.all import Graph
from typing import Tuple
from paulis import Pauli
from numpy.typing import NDArray


class Hamiltonian:
    def __init__(self, weights: list[float], ops: list[Pauli]):
        self.num_ops = len(ops)
        assert self.num_ops == len(weights)
        if self.num_ops == 0:
            print("Warning: Hamiltonian has no terms; setting spin number to 1!")
            self.n = 1
        else:
            self.n = ops[0].n
        self.weights = weights
        self.ops = ops

    @classmethod
    def safe_init(cls, weights: list[float], ops: list[Pauli]) -> "Hamiltonian":
        has_prop, _ = has_proportional_terms([p for p in ops])
        if has_prop:
            raise ValueError("Hamiltonian has proportional terms")
        n = ops[0].n
        for p in ops:
            if p.n != n:
                raise ValueError(
                    "All Pauli operators must have the same number of qubits"
                )
        return Hamiltonian(weights, ops)

    def get_frustration_graph(self) -> Tuple[Graph, list[float]]:
        """Return the frustration graph with the vertex weights."""
        g = Graph()
        for i in range(self.num_ops):
            g.add_vertex(i)
        for i in range(self.num_ops):
            for j in range(i + 1, self.num_ops):
                if self.ops[i].symplectic_inner_product(self.ops[j]):
                    g.add_edge(i, j)
        return g, self.weights

    def to_matrix(self) -> NDArray[np.complex128]:
        mat = np.zeros((2**self.n, 2**self.n), dtype=np.complex128)
        for w, op in zip(self.weights, self.ops):
            mat += w * op.to_matrix()
        return mat


def has_proportional_terms(
    ops: list[Pauli],
) -> Tuple[bool, None | list[Tuple[int, int]]]:
    """Check if there are proportional terms in the list of Paulis."""
    proportional_pairs: list[Tuple[int, int]] = []
    for i in range(len(ops)):
        for j in range(i + 1, len(ops)):
            if ops[i].is_proportional_to(ops[j]):
                proportional_pairs.append((i, j))
    if len(proportional_pairs) > 0:
        return True, proportional_pairs
    else:
        return False, None
