from enum import Enum

import numpy as np
from sage.all import Graph
from typing import Tuple

from rust_backend.hitting import (
    HitterU8,
    HitterU16,
    HitterU32,
    HitterU64,
    HitterU128,
    HitterUsize,
)


class HitterPy:
    def __init__(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)

    def hit(
        self,
        sets_to_hit: list[list[int]],
        weights: list[float],
        randomise: bool = False,
    ) -> set[int]:
        """Approximation algorithm for the hitting set problem.

        Let X = {0, ..., n-1}, n in N, be a set of n elements, with according `weights`,
        and `sets_to_hit` be a list of sets of elements from X. This function calculates a
        minimal hitting set, according to Bar-Yehuda's greedy algorithm.

        Args:
            sets_to_hit (list[list[int]]): The list of sits that have to be hit.
            mut weights (list[float]): The weights for each element in X.
        Returns:
            set[int]: The calculated minimal hitting set.
        """
        solution = set()
        to_hit = set(range(len(sets_to_hit)))

        if randomise:
            sets_to_hit = sets_to_hit.copy()
            self.rng.shuffle(sets_to_hit)

        while len(to_hit) > 0:
            # the next(iter(to_hit)) is apparently deterministic in python ..., so we are
            # indeed deterministic here if randomise=False
            pricing_set = sets_to_hit[next(iter(to_hit))]
            pricing_weights = [weights[i] for i in pricing_set]
            min_element, min_weight = min(
                zip(pricing_set, pricing_weights), key=lambda x: x[1]
            )
            for e in pricing_set:
                weights[e] -= min_weight
            solution.add(min_element)
            to_hit -= {i for i in to_hit if min_element in sets_to_hit[i]}

        solution_ret = solution.copy()
        for e in solution:
            solution_reduced = solution_ret.copy()
            solution_reduced.remove(e)
            hit_all = True
            for set_to_hit in sets_to_hit:
                if not any(e in solution_reduced for e in set_to_hit):
                    hit_all = False
                    break
            if hit_all:
                solution_ret = solution_reduced

        return solution_ret


class RustUnsignedInteger(Enum):
    U8 = 0
    U16 = 1
    U32 = 2
    U64 = 3
    U128 = 4
    Usize = 5
    # original version against which I tested (a little bit) the Rust implementation
    SlowPythonImplementation = 6


class Hitter:
    def __init__(
        self,
        seed: int | None = None,
        integer_type: RustUnsignedInteger = RustUnsignedInteger.Usize,
    ):
        self._seed = seed
        self._integer_type = integer_type
        self.update_hitter()

    @property
    def integer_type(self) -> RustUnsignedInteger:
        return self._integer_type

    @property
    def seed(self) -> int | None:
        return self._seed

    def update_hitter(
        self,
        seed: int | None | bool = False,
        integer_type: RustUnsignedInteger | None = None,
    ):
        """
        seed = False means 'do not update the seed'; a seed of None is equivalent to a
        random seed
        """
        if seed is None or type(seed) == int:
            self._seed = seed
        if integer_type is not None:
            self._integer_type = integer_type
        match self._integer_type:
            case RustUnsignedInteger.U8:
                self.hitter = HitterU8(self.seed)
            case RustUnsignedInteger.U16:
                self.hitter = HitterU16(self.seed)
            case RustUnsignedInteger.U32:
                self.hitter = HitterU32(self.seed)
            case RustUnsignedInteger.U64:
                self.hitter = HitterU64(self.seed)
            case RustUnsignedInteger.U128:
                self.hitter = HitterU128(self.seed)
            case RustUnsignedInteger.Usize:
                self.hitter = HitterUsize(self.seed)
            case RustUnsignedInteger.SlowPythonImplementation:
                self.hitter = HitterPy(self.seed)

    # {{{ I'm making sure the types of seed and integer_type are correct with some
    # assertions because I'm not sure what python does if I set them to the wrong type
    # (e.g., setting seed to False causes False being passed to the actual Hitter as seed
    # and python just runs with that, converting it probably to 0, but that feels
    # uncontrolled to me)
    @integer_type.setter
    def integer_type(self, integer_type: RustUnsignedInteger):
        assert type(integer_type) == RustUnsignedInteger
        self._integer_type = integer_type
        self.update_hitter(integer_type=integer_type)

    @seed.setter
    def seed(self, seed: int | None):
        assert seed is None or type(seed) == int
        self._seed = seed
        self.update_hitter(seed=seed)
    # }}}


    def hit(
        self,
        sets_to_hit: list[list[int]],
        weights: list[float],
        randomise: bool = False,
    ) -> set[int]:
        return self.hitter.hit(sets_to_hit, weights, randomise)


# Note on the "PERF" comments: things run much faster when making use of max_sets_to_hit
# (cf. LineGraphSearch.search), at the cost of less optimal results. I think with an
# appropriate number for max_sets_to_hit (it can be quite high!), this is a good trade-off
# (definitely for testing!).
def subgraph_search(
    g: Graph,
    vertex_weights: dict,
    forbidden_graphs: list[Tuple[int, list[Graph]]],  # [(size, [graphs of that size])]
    hitter: Hitter,
    randomise: bool = False,
    max_sets_to_hit: int | None = None,
) -> Tuple[Graph, list[int], int]:
    raw_sets_to_hit = []
    # PERF: this here is another bottleneck, but not as bad as the hitting step (depending
    # on "max_sets_to_hit")
    print("getting sets to hit ...")
    if max_sets_to_hit is None:
        for _, fgs in forbidden_graphs:
            # this method gives us isomorphic duplicates
            # (https://github.com/sagemath/sage/issues/35821)
            for fg in fgs:
                raw_sets_to_hit.extend(
                    g.subgraph_search_iterator(fg, induced=True, return_graphs=False)
                )
                # filtering makes the loop here so slow that even though the actual
                # hitting step should be faster, the overall runtime is worse
                # seen = set()
                # for subgraph in g.subgraph_search_iterator(
                #     fg, induced=True, return_graphs=True
                # ):
                #     subgraph = tuple(sorted(subgraph))
                #     if subgraph in seen:
                #         continue
                #     seen.add(subgraph)
                #     raw_sets_to_hit.append(subgraph)
            # with this method we do not get isomorphic duplicates, but it is very, very,
            # very slow ...
            # for sub_vertices in g.connected_subgraph_iterator(
            #     k=k, vertices_only=True, exactly_k=True
            # ):
            #     for fg in fgs:
            #         if g.subgraph(sub_vertices).is_isomorphic(fg):  # pyright: ignore
            #             raw_sets_to_hit.append([v for v in sub_vertices])
    else:
        counter = 0
        breaking = False
        for _, fgs in forbidden_graphs:
            for fg in fgs:
                for set_to_hit in g.subgraph_search_iterator(
                    fg, induced=True, return_graphs=False
                ):
                    counter += 1
                    raw_sets_to_hit.append(set_to_hit)
                    if counter >= max_sets_to_hit:
                        breaking = True
                        break
                if breaking:
                    break
            if breaking:
                break
    print("got sets")

    num_sets_to_hit = len(raw_sets_to_hit)
    if num_sets_to_hit == 0:
        return g, [], 0

    # hitting.hitting requires the vertices to be labeled from 0 to n-1, so we have to
    # do some mapping
    cover = set()
    for set_to_hit in raw_sets_to_hit:
        for v in set_to_hit:
            cover.add(v)

    cover_map = list(cover)
    inv_cover_map = {v: i for i, v in enumerate(cover_map)}

    weights = []
    for v in cover_map:
        # we minimise w.r.t. the L^1 norm (w.r.t. a Pauli basis) which is an upper
        # bound of the operator norm
        weights.append(np.abs(vertex_weights[v]))
    sets_to_hit = []
    for raw_set_to_hit in raw_sets_to_hit:
        set_to_hit = []
        for v in raw_set_to_hit:
            set_to_hit.append(inv_cover_map[v])
        sets_to_hit.append(set_to_hit)

    # PERF: this is the bottleneck
    # PERF: write in Rust -> still the bottleneck for large "sets_to_hit"; but if not too
    # large the bottleneck above is worse
    print("hitting ...")
    mapped_hitting_set = hitter.hit(sets_to_hit, weights, randomise)
    print("got hitting set")
    hitting_set = [cover_map[i] for i in mapped_hitting_set]

    ret = g.copy()
    ret.delete_vertices([v for v in g.vertices() if v in hitting_set])
    return ret, hitting_set, num_sets_to_hit
