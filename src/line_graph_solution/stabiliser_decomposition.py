import numpy as np
from numpy.typing import NDArray

import paulis
from paulis import Pauli
from hamiltonian import Hamiltonian
from line_graph_solution.mapping import Mapping
from linalg.f2 import matrix


def get(
    mapping: Mapping,
    lamda_matrix: NDArray[np.float64],
    hamiltonian: Hamiltonian,
):
    num_majoranas = lamda_matrix.shape[0]

    for i in range(0, num_majoranas):
        for j in range(0, num_majoranas):
            if np.abs(lamda_matrix[i, j]) < 1e-13:
                lamda_matrix[i, j] = 0.0

    lamda_pairs = []
    i = 0
    while i < num_majoranas:
        for j in range(num_majoranas):
            x = lamda_matrix[i, j]
            if x != 0.0:
                assert np.isclose(x, -lamda_matrix[i + 1, j - 1])
                lamda_pairs.append(((i, i + 1), x))
                i += 1
                break
        i += 1

    # we want them sorted so that the dependent stabilisers have rather small lamdas ->
    # should give better results with heuristics when trying to find a sign choice for the
    # independent stabilisers
    lamda_pairs.sort(key=lambda pair: abs(pair[1]))

    lamda_stabilisers: list[Pauli] = []
    for i, pair in enumerate(lamda_pairs):
        s = mapping.psi_to_pauli(pair[0][0], pair[0][1], hamiltonian)
        # we put signs into Paulis to make all lamdas positive
        if np.sign(pair[1]) == -1:
            s.phase = (s.phase + 2) % 4
            lamda_pairs[i] = (pair[0], -pair[1])
        lamda_stabilisers.append(s)

    all_stabilisers = mapping.sym_generators + lamda_stabilisers

    all_independent_stabilisers: list[Pauli] = matrix.max_independent_set(
        all_stabilisers  # pyright: ignore
    )
    lamda_independent_stabilisers = all_independent_stabilisers[
        mapping.num_symmetry_generators :
    ]
    all_independent_indices = [
        all_stabilisers.index(s) for s in all_independent_stabilisers
    ]
    i = 0
    while i < mapping.num_symmetry_generators:
        assert all_independent_indices[i] == i
        i += 1
    lamda_independent_indices = [
        lamda_stabilisers.index(s) for s in lamda_independent_stabilisers
    ]

    lamda_identites = []
    lamda_identites_indices = []
    lamda_dependent_stabilisers = []
    lamda_dependent_indices = []
    for i, s in enumerate(lamda_stabilisers):
        if s.is_proportional_to_identity():
            assert s not in lamda_independent_stabilisers
            assert s.phase % 2 == 0
            lamda_identites.append(s)
            lamda_identites_indices.append(i)
        elif s not in lamda_independent_stabilisers:
            lamda_dependent_stabilisers.append(s)
            lamda_dependent_indices.append(i)

    merged_dependent_decompositions: list[tuple[int, list[int]]] = [  # pyright: ignore
        paulis.decompose_into_independent_paulis(all_independent_stabilisers, s)
        for s in lamda_dependent_stabilisers
    ]

    splitted_dependent_decompositions = []
    for decomposition in merged_dependent_decompositions:
        splitted_dependent_decomposition = (decomposition[0], [], [])
        for i in decomposition[1]:
            if i < mapping.num_symmetry_generators:
                splitted_dependent_decomposition[1].append(i)
            else:
                splitted_dependent_decomposition[2].append(
                    i - mapping.num_symmetry_generators
                )
        splitted_dependent_decompositions.append(splitted_dependent_decomposition)

    return (
        lamda_pairs,
        all_stabilisers,
        lamda_stabilisers,
        all_independent_stabilisers,
        lamda_independent_stabilisers,
        lamda_dependent_stabilisers,
        all_independent_indices,
        lamda_independent_indices,
        lamda_dependent_indices,
        merged_dependent_decompositions,
        splitted_dependent_decompositions,
        lamda_identites,
        lamda_identites_indices,
    )
