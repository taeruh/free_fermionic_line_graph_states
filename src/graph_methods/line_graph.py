from sage.all import Graph
from sage.all import graphs  # pyright: ignore  (this is sage.graphs ...)
from sage.graphs import line_graph  # pyright: ignore
from typing import Tuple

from . import hitting, djp_tree


class LineGraphSearch:
    def __init__(self, seed: int | None = None, cycle_free: tuple[bool, int] = (False, 0)):
        self.hitter = hitting.Hitter(seed)
        if cycle_free[0]:
            self.forbidden_graphs = []
            # at least 6 so that the forbidden_subgraph loop below doesn't error
            max_cycle_size = max(6, cycle_free[1])
            for size in range(4, max_cycle_size + 1):
                self.forbidden_graphs.append((size, [graphs.CycleGraph(size)]))
        else:
            self.forbidden_graphs = [(4, []), (5, []), (6, [])]
        for g in graphs.line_graph_forbidden_subgraphs():
            n = g.num_verts()
            assert 4 <= n <= 6
            # if we don't randomise the hitting, I think it is better if the hitting first
            # removes the forbidden graphs because there are likely many more cycles
            # (which are then hopefully already partially removed)?
            self.forbidden_graphs[n - 4][1].insert(0, g)
            # self.forbidden_graphs[n - 4][1].append(g)

    def search(
        self,
        g,
        vertex_weights: dict,
        randomise: bool = False,
        max_sets_to_hit: int | None = None,
    ) -> Tuple[Graph, list[int], int]:
        if max_sets_to_hit is None:
            return hitting.subgraph_search(
                g,
                vertex_weights,
                self.forbidden_graphs,
                self.hitter,
                randomise,
                max_sets_to_hit,
            )
        else:
            linegraph = g.copy()
            hitting_set = []
            total_num_hitted = 0
            current_num_hitted = 1
            while current_num_hitted > 0:
                print("hitted set:", current_num_hitted)
                print("remaning vertiecs:", linegraph.num_verts())
                linegraph, hitting_set, current_num_hitted = hitting.subgraph_search(
                    linegraph,
                    vertex_weights,
                    self.forbidden_graphs,
                    self.hitter,
                    randomise,
                    max_sets_to_hit,
                )
                total_num_hitted += current_num_hitted
            return linegraph, hitting_set, total_num_hitted


def is_line_graph(g: Graph) -> bool:
    return line_graph.is_line_graph(g)


def get_root_graph(g: Graph) -> Tuple[Graph, dict]:
    """
    Returns the root graph of the line graph g, and the isomorphism mapping vertices in
    g to edges in the root graph.
    """
    return line_graph.root_graph(g)


def get_line_subgraph(
    graph: Graph,
    weights: list[float],
    randomise: bool = False,
    seed: int | None = None,
    max_sets_to_hit: int | None = None,
    # TODO: add option to make it cycle free after the linegraph hitting by finding the
    # maximum spanning tree -> compare which method gives better results
    cycle_free: bool = False,
) -> Tuple[Graph, Graph, Graph, dict, list[int], list[Graph], list[Graph]]:
    weights_dict = {i: weights[i] for i in range(len(weights))}

    linegraph = None
    if cycle_free:
        linegraph, _, _ = LineGraphSearch(
            seed, cycle_free=(True, graph.num_verts())
        ).search(graph, weights_dict, randomise, max_sets_to_hit)
    elif not is_line_graph(graph):
        linegraph, _, _ = LineGraphSearch(seed).search(
            graph, weights_dict, randomise, max_sets_to_hit
        )
    else:
        print("graph is already a line graph, skipping hitting")
        linegraph = graph.copy()
    assert is_line_graph(linegraph)

    components = [
        linegraph.subgraph(vertex_set)
        for vertex_set in linegraph.connected_components()
    ]
    root_graph = Graph()
    root_tree = Graph()
    root_graphs = []
    root_trees = []
    full_isomorphism = {}
    root_sizes = []
    for i, component in enumerate(components):
        root, isom = get_root_graph(component)
        for v, e in isom.items():
            root.set_edge_label(e[0], e[1], weights[v])
        tree = djp_tree.maximum_spanning_tree(root)
        # need unique labels, otherwise merging the components together later makes
        # problems (after the merge, we relabel again to nicer labels)
        component_label_map = lambda x: f"{x}_c{i}"
        root.relabel(component_label_map)
        tree.relabel(component_label_map)
        root_graph = root_graph.union(root)
        root_tree = root_tree.union(tree)
        root_graphs.append(root)
        root_trees.append(tree)
        for v, e in isom.items():
            assert full_isomorphism.get(v) is None
            full_isomorphism[v] = (component_label_map(e[0]), component_label_map(e[1]))
        root_sizes.append(root.num_verts())

    label_map = {v: i for i, v in enumerate(root_graph.vertices())}
    root_graph.relabel(label_map, inplace=True)
    root_tree.relabel(label_map, inplace=True)
    for root, tree in zip(root_graphs, root_trees):
        root.relabel(label_map, inplace=True)
        tree.relabel(label_map, inplace=True)
    for v, e in full_isomorphism.items():
        full_isomorphism[v] = (label_map[e[0]], label_map[e[1]])

    return (
        linegraph,
        root_graph,
        root_tree,
        full_isomorphism,
        root_sizes,
        root_graphs,
        root_trees,
    )
