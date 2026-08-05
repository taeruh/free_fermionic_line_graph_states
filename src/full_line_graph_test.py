import numpy as np

from linalg.f2 import matrix
from hamiltonian import Hamiltonian
import paulis
from paulis import Pauli
import clifford
from graph_methods import line_graph
from line_graph_solution import (
    cycles,
    majorana_diagonalisation,
    matchgate_decomposition,
    euler_hoffman_decomposition,
    stabiliser_decomposition,
)
from line_graph_solution.mapping import Mapping
from models.heisenberg_2d import Heisenberg2D, Interaction
from models import RandomCoupling, ConstantCoupling
from hamiltonian import Hamiltonian


def run():
    seed = 2
    # seed = None
    h = Heisenberg2D(
        3,
        2,
        Interaction(
            RandomCoupling(-2, 5, seed),
            # RandomCoupling(-3, 2, seed),
            # RandomCoupling(-1, 1, seed),
            # RandomCoupling(-1, 1, seed),
            ConstantCoupling(0),
            ConstantCoupling(0),
            ConstantCoupling(0),
            RandomCoupling(-5, 1, seed),
            # RandomCoupling(-3, 5, seed),
            # RandomCoupling(-1, 1, seed),
            # RandomCoupling(-1, 1, seed),
            ConstantCoupling(0),
            ConstantCoupling(0),
            ConstantCoupling(0),
            RandomCoupling(-9, 7, seed),
        ),
        0,
        False,
    )
    test(h.hamiltonian)


def test(hamiltonian: Hamiltonian) -> None:
    graph, weights = hamiltonian.get_frustration_graph()

    total_weight = sum([np.abs(w) for w in weights])

    (
        linegraph,
        root_graph,
        root_tree,
        full_isomorphism,
        root_sizes,
        root_graphs,
        root_trees,
    ) = line_graph.get_line_subgraph(graph, weights, cycle_free=True)
    full_isomorphism_inverse = {v: k for k, v in full_isomorphism.items()}

    line_vertices = set(linegraph.vertices())
    line_weight = sum([np.abs(weights[v]) for v in line_vertices])

    perturbation_vertices = set(graph.vertices()) - line_vertices
    perturbation_weight = sum([np.abs(weights[v]) for v in perturbation_vertices])

    mapping = Mapping(
        root_graph,
        full_isomorphism,
        full_isomorphism_inverse,
        cycles.get_cycles(root_graphs, root_trees, full_isomorphism_inverse),
        hamiltonian,
    )

    num_symmetry_generators = mapping.num_symmetry_generators
    dim_hilbert_space = 2**hamiltonian.n
    num_majoranas = root_graph.num_verts()

    line_hamiltonian = Hamiltonian(
        [hamiltonian.weights[i] for i in line_vertices],
        [hamiltonian.ops[i] for i in line_vertices],
    )

    line_matrix = line_hamiltonian.to_matrix()

    line_matrix_reconstruction_sum = np.zeros(
        (dim_hilbert_space, dim_hilbert_space), dtype=np.complex128
    )
    projector_sum = np.zeros((dim_hilbert_space, dim_hilbert_space))

    for sector_code in range(2**num_symmetry_generators):
        symmetry_sector = [0] * num_symmetry_generators
        for z_idx in range(num_symmetry_generators):
            if (sector_code >> z_idx) & 1:
                symmetry_sector[z_idx] = 1

        mapping.update_symmetry_sector(symmetry_sector)

        projector = np.identity(dim_hilbert_space)
        for phase, generator in zip(symmetry_sector, mapping.sym_generators):
            projector = (
                projector
                @ (
                    np.identity(dim_hilbert_space)
                    + (-1) ** phase * generator.to_matrix()
                )
                / 2
            )
        # the first two asserts are rather checking than the generators were correctly
        # defined... if they are, it is clear that these two asserts hold
        assert np.allclose(projector @ projector, projector)
        assert np.allclose(projector.conj().T, projector)
        assert np.allclose(projector @ line_matrix, line_matrix @ projector)
        projector_sum = projector_sum + projector

        majorana_matrix = majorana_diagonalisation.get_h(mapping, weights)
        assert majorana_matrix.shape == (num_majoranas, num_majoranas)
        reconstruct_line_matrix = np.zeros(line_matrix.shape, dtype=np.complex128)
        for z_idx in range(majorana_matrix.shape[0]):
            for j in range(majorana_matrix.shape[1]):
                if majorana_matrix[z_idx, j] != 0.0:
                    reconstruct_line_matrix += (
                        majorana_matrix[z_idx, j]
                        * mapping.psi_to_pauli(z_idx, j, hamiltonian).to_matrix()
                    )
        assert np.allclose(line_matrix, reconstruct_line_matrix)

        lamda_matrix, k_matrix = majorana_diagonalisation.skew_diagonalise(
            majorana_matrix, root_sizes
        )

        (
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
        ) = stabiliser_decomposition.get(mapping, lamda_matrix, hamiltonian)

        for s in lamda_stabilisers:
            s_matrix = s.to_matrix()
            assert np.allclose(s_matrix @ projector, projector @ s_matrix, 0.0, 0.0)

        _, _, full_clifford_matrix = clifford.get_clifford_from_paulis(
            all_independent_stabilisers
        )

        # start to reconstruct the whole hamiltonian from our majorana solution (we don't
        # have a simple operator to diagonalise the hamiltonian since the projectors do
        # not commute with the clifford preparotion, however, we can do the inverse of the
        # diagonalisation to get the hamiltonian back from the diagonal form)
        line_matrix_reconstruction = np.zeros(line_matrix.shape, dtype=np.complex128)

        stabiliser_matrix_check = np.zeros(line_matrix.shape, dtype=np.complex128)

        for lamda_idx, stab in zip(
            lamda_independent_indices, lamda_independent_stabilisers
        ):
            lambda_val = lamda_pairs[lamda_idx][1]
            z_idx = all_independent_stabilisers.index(stab)
            z_pauli = Pauli.identity(hamiltonian.n)
            z_pauli.u[z_idx] = True
            stabiliser_matrix_check += 2 * lambda_val * stab.to_matrix()
            line_matrix_reconstruction += 2 * lambda_val * z_pauli.to_matrix()
        for lamda_idx, stab, decomposition in zip(
            lamda_dependent_indices,
            lamda_dependent_stabilisers,
            merged_dependent_decompositions,
        ):
            lambda_val = lamda_pairs[lamda_idx][1]
            phase = decomposition[0]
            # stab product here just to test the decomposition again
            stab_product = Pauli.identity(hamiltonian.n)
            z_product = Pauli.identity(hamiltonian.n)
            for z_idx in decomposition[1]:
                stab_product = stab_product.multiply_as_paulis(
                    all_independent_stabilisers[z_idx]
                )
                z_pauli = Pauli.identity(hamiltonian.n)
                z_pauli.u[z_idx] = True
                z_product = z_product.multiply_as_paulis(z_pauli)
            stab_product.phase = (stab_product.phase + phase) % 4
            z_product.phase = (z_product.phase + phase) % 4
            assert stab_product.multiply_as_paulis(stab).is_identity()
            stabiliser_matrix_check += 2 * lambda_val * stab.to_matrix()
            line_matrix_reconstruction += 2 * lambda_val * z_product.to_matrix()

        # now, line_matrix_reconstruction, is the diagonalised hamiltonian

        # now, line_matrix_reconstruction, is in the form of the stabilisers
        line_matrix_reconstruction = (
            full_clifford_matrix
            @ line_matrix_reconstruction
            @ full_clifford_matrix.conj().T
        )
        assert np.allclose(line_matrix_reconstruction, stabiliser_matrix_check)

        # the projector does not necessarily commute with the clifford preparation, but it
        # does commute with the stabilisers
        assert np.allclose(
            line_matrix_reconstruction @ projector,
            projector @ line_matrix_reconstruction,
        )

        angles = euler_hoffman_decomposition.checked_get_angles_and_indices(
            k_matrix, root_sizes
        )
        exponents = matchgate_decomposition.get_exponent_sequence(
            angles, mapping, hamiltonian
        )

        matchgates = matchgate_decomposition.get_matchgates(exponents)
        matchgate = matchgates[0]
        for gate in matchgates[1:]:
            matchgate = matchgate @ gate
        assert np.allclose(
            np.identity(dim_hilbert_space), matchgate @ matchgate.conj().T
        )
        assert np.allclose(projector @ matchgate, matchgate @ projector)

        line_matrix_reconstruction = (
            matchgate.conj().T @ line_matrix_reconstruction @ matchgate
        )
        assert np.allclose(
            line_matrix_reconstruction @ projector,
            projector @ line_matrix_reconstruction,
        )

        line_matrix_reconstruction = projector @ line_matrix_reconstruction
        line_matrix_projected = line_matrix @ projector
        if not np.allclose(line_matrix_projected, line_matrix_reconstruction):
            print("block inverse diagonalisation fail")
            err = np.sum(
                np.abs(line_matrix_projected - line_matrix_reconstruction).flatten()
            )
            norm = np.sum(np.abs(line_matrix_projected).flatten())
            rel_err = err / norm
            if rel_err < 1e-5:
                print(f"...but acceptable relative error ({rel_err})")
            else:
                raise ValueError(
                    f"too large relative error ({rel_err}); error of {err}"
                    f"to norm of {norm}"
                )

        line_matrix_reconstruction_sum += line_matrix_reconstruction

    assert np.allclose(projector_sum, np.identity(dim_hilbert_space), 0.0, 0.0)

    if not np.allclose(line_matrix, line_matrix_reconstruction_sum):
        print("inverse diagonalisation fail")
        err = np.sum(np.abs(line_matrix - line_matrix_reconstruction_sum).flatten())
        norm = np.sum(np.abs(line_matrix).flatten())
        rel_err = err / norm
        if rel_err < 1e-5:
            print(f"...but acceptable relative error ({rel_err})")
        else:
            raise ValueError(
                f"too large relative error ({rel_err}); error of {err}"
                f"to norm of {norm}"
            )

    print("Test passed!")
