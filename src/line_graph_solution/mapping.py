from sage.all import Graph
from typing import Tuple

import paulis
from paulis import Pauli
from linalg.f2 import matrix
from linalg.f2.matrix import Matrix
from hamiltonian import Hamiltonian


class BaseMapping:
    def __init__(self, root: Graph, isom: dict):
        self.root = root
        self.isom = isom
        self.num_modes = root.num_verts()
        # in the phi and psi_gen mappings we implicitly include the "i" factor in
        # g mapsto i gamma_u gamma_v !!!!!!
        self.phi = {}  # has to be filled in by subclasses
        self.psi_gen = {}  # has to be filled in by subclasses

    def _init_phi_base(self):
        self.phi = self.isom.copy()
        # first tuple element is d(u, v); d=0 zero here is potentially just a placeholder
        # and has to be filled in by subclasses
        for v, e in self.phi.items():
            self.phi[v] = (0, e)

    # note that while we define psi via the tree in our notes/paper, we can also include
    # the missing cycle edges here into psi/psi_gen; it's not a minimal generating set
    # anymore than but still a generating set and we might get more straightforward paths
    # in the psi() function below; calculating psi on those edges via the remaining cycle
    # would be coherent with the definition in the notes/paper, however, a much faster way
    # is to just use that psi inverts phi on ALL edges, i.e., especially also on those
    # missing cycle edges
    def _init_psi_gen_base(self):
        for v, (s, e) in self.phi.items():
            self.psi_gen[(e[0], e[1])] = (s, v)
            self.psi_gen[(e[1], e[0])] = ((s + 1) % 2, v)

    def psi(self, u: int, v: int) -> Tuple[int, list[int]]:
        """
        input: two vertices u, v in the root graph for which we want to find out to what
        i gamma_u gamma_v maps to in the line graph (note the "i" we implicitly include
        here!)

        returns (b, list[v]) where i^b, with b in {0, 1, 2, 3}, is a global phase
        due to the length of the path from u to v (collect multiple "i"s), and the
        direction of the edges; and list[v] is the corresponding vertex list in the
        line graph
        """
        # cf. the note above _init_psi_gen_base on why we take paths in root and not tree;
        # here we see the two advantages of including those missing cycle edges: firstly,
        # the paths might be more straightforward, and secondly, we don't need the tree
        # here at all, i.e., we can keep the code simpler (our mapping classes don't
        # required the tree anymore!)
        path = next(self.root.all_paths_iterator([u], [v], simple=True))
        num_edges = len(path) - 1
        # number of extra edges (first gives the "i" we encoded into the input)
        # TODO: check the correctness of the "* 3" here again (I think before the break, I
        # convinced myself that it has to be "* 1", but now I convinced myself that it has
        # to be "* 3"...)
        b = ((num_edges - 1) * 3) % 4
        vertex_list = []
        for i in range(num_edges):
            edge = (path[i], path[i + 1])
            d, v = self.psi_gen[edge]
            b = (b + 2 * d) % 4
            vertex_list.append(v)
        return (b, vertex_list)

    def psi_to_pauli(self, u: int, v: int, h: Hamiltonian) -> Pauli:
        b, vertex_list = self.psi(u, v)
        p = h.ops[vertex_list[0]].clone()
        for vertex in vertex_list[1:]:
            p = p.multiply_as_paulis(h.ops[vertex])
        p.phase = (p.phase + b) % 4
        return p


class NoSymmetriesMapping(BaseMapping):
    def __init__(self, root: Graph, isom: dict):
        """
        it is assumed that the root graph has no cycles; assume that
        isom is also restricted to the root graph; assume the vertices in the root graph
        are labeled from 0 to n-1;
        """
        super().__init__(root, isom)
        self._init_phi_base()
        self._init_psi_gen_base()


class Mapping(BaseMapping):
    def __init__(
        self,
        root: Graph,
        isom: dict,
        isom_inv: dict,
        cycle_paths: list[Tuple[Tuple[int, int], list[Tuple[int, int]]]],
        hamiltonian: Hamiltonian,
    ):
        super().__init__(root, isom)
        self._init_phi_base()
        self._init_psi_gen_base()

        self.isom_inv = isom_inv
        self.cycle_paths = cycle_paths

        self.symmetries: list[Pauli] = []
        for _, cycle in cycle_paths:
            vertices = [isom_inv[edge] for edge in cycle]
            pauli_ops = [hamiltonian.ops[v].clone() for v in vertices]
            sym: Pauli = pauli_ops[0].clone()
            for p in pauli_ops[1:]:
                sym = sym.multiply_as_paulis(p)
            if len(vertices) % 2 == 1:
                sym.phase = (sym.phase + 1) % 4
            self.symmetries.append(sym)
        self.sym_identities = []
        for s in self.symmetries:
            if s.is_proportional_to(Pauli.identity(s.n)):
                self.sym_identities.append(s)
        self.sym_identities_to_cycle_index = []
        for g in self.sym_identities:
            self.sym_identities_to_cycle_index.append(self.symmetries.index(g))
        remaining_syms = []
        for i, s in enumerate(self.symmetries):
            if i not in self.sym_identities_to_cycle_index:
                remaining_syms.append(s)
        self.sym_generators: list[Pauli] = matrix.max_independent_set(
            remaining_syms  # pyright: ignore
        )
        self.sym_generators_to_cycle_index = []
        for g in self.sym_generators:
            self.sym_generators_to_cycle_index.append(self.symmetries.index(g))
        self.sym_other = []
        self.sym_other_to_cycle_index = []
        for i, s in enumerate(self.symmetries):
            if (
                i not in self.sym_identities_to_cycle_index
                and i not in self.sym_generators_to_cycle_index
            ):
                self.sym_other.append(s)
                self.sym_other_to_cycle_index.append(i)
        # that will be the k(s) and y(s) for s in sym_other (cf. notes)
        self.sym_other_decompositions = []
        for other in self.sym_other:
            other_phase, other_decomposition = paulis.decompose_into_independent_paulis(
                self.sym_generators, other
            )
            k = 0
            if other_phase == 0:
                k = 0
            elif other_phase == 2:
                k = 1
            else:
                raise ValueError(
                    f"""symmetry decomposition failed; the additional phase should be
                    either 0 or 2, but got {other_phase}"""
                )
            self.sym_other_decompositions.append((k, other_decomposition))
        # symmetries proportional to identities are always chosen to be +1, i.e., they
        # don't give us freedom but they also do not depend on the generator choices
        for i, sym in enumerate(self.sym_identities):
            assert sym.phase % 2 == 0
            sign = sym.phase // 2
            fixed_edge, cycle = self.cycle_paths[self.sym_identities_to_cycle_index[i]]
            d = (sign + self._update_symmetry_sector_helper(fixed_edge, cycle)) % 2
            self.phi[self.isom_inv[fixed_edge]] = (d, fixed_edge)

        self.num_symmetry_generators = len(self.sym_generators)

        # do a first symmetry sector initalisiation
        self.symmetry_sector = [0] * self.num_symmetry_generators
        self.update_symmetry_sector(self.symmetry_sector)

    def update_symmetry_sector(self, x: list[int]):
        assert len(x) == self.num_symmetry_generators
        self.symmetry_sector = x.copy()

        for i in range(self.num_symmetry_generators):
            fixed_edge, cycle = self.cycle_paths[self.sym_generators_to_cycle_index[i]]
            d = (x[i] + self._update_symmetry_sector_helper(fixed_edge, cycle)) % 2
            self.phi[self.isom_inv[fixed_edge]] = (d, fixed_edge)
        for i in range(len(self.sym_other)):
            k, decom = self.sym_other_decompositions[i]
            d = k
            for j in decom:
                d += x[j]
            fixed_edge, cycle = self.cycle_paths[self.sym_other_to_cycle_index[i]]
            d = (d + self._update_symmetry_sector_helper(fixed_edge, cycle)) % 2
            self.phi[self.isom_inv[fixed_edge]] = (d, fixed_edge)

        for edge, _ in self.cycle_paths:
            v = self.isom_inv[edge]
            sign, same_edge = self.phi[v]
            assert edge == same_edge
            self.psi_gen[(edge[0], edge[1])] = (sign, v)
            self.psi_gen[(edge[1], edge[0])] = ((sign + 1) % 2, v)

    def _update_symmetry_sector_helper(self, fixed_edge, cycle) -> int:
        d = 0
        flattened_cycle = []
        for e in cycle:
            flattened_cycle.append(e[0])
            flattened_cycle.append(e[1])
        d += sort_majorana_cycle(flattened_cycle)
        cycle_len = len(cycle)
        num_is = ((cycle_len % 2) + cycle_len) % 4
        assert num_is % 2 == 0
        d += num_is // 2
        for edge in cycle:
            if edge == fixed_edge:
                continue
            sign = self.phi[self.isom_inv[edge]][0]
            assert sign == 0  # test: we are never changing them
            # if we would change the sign, we would add it to d here
            # d += sign
        return d % 2


def sort_majorana_cycle(cycle: list[int]) -> int:
    """
    sort the cycle such that everything cancels and return the sign (the actual sign is
    (-1)^sign)
    """
    n = len(cycle)
    sign = 0
    for i in range(n):
        for j in range(i + 1, n):
            if cycle[i] > cycle[j]:
                for k in range(j, i, -1):
                    cycle[k], cycle[k - 1] = cycle[k - 1], cycle[k]
                    # the following can actually never not happen, because we already
                    # would have swapped the element at k-1 to the left
                    # if cycle[k] != cycle[k - 1]:
                    #     sign += 1
                    assert cycle[k] != cycle[k - 1]
                    sign += 1
    return sign % 2

    # print("START")
    # print(self.symmetries)
    # print(self.sym_identities)
    # print(self.sym_generators)
    # print(self.sym_other)
    # print(self.sym_identities_to_cycle_index)
    # print(self.sym_generators_to_cycle_index)
    # print(self.sym_other_to_cycle_index)
    # print([isom_inv[edge] for edge, _ in self.cycle_paths])
    # print("END")
