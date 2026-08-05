from sage.all import Graph
from typing import Tuple


def _get_unique_path_to_root(
    tree: Graph, r: int, v: int, isom_inv: dict
) -> list[Tuple[int, int]]:
    """we direct the edges according to isom and isom_inv, i.e., these mappings only
    contain one direction of each edge"""
    paths = tree.all_paths(r, v)
    assert len(paths) == 1
    path = paths[0]
    edge_path = []
    for i in range(len(path) - 1):
        edge = (path[i], path[i + 1])
        order_label = isom_inv.get(edge)
        if order_label is None:
            edge = (edge[1], edge[0])
            order_label = isom_inv.get(edge)
        assert order_label is not None
        edge_path.append(edge)
    return edge_path


def _get_symmetry_cycle(
    tree: Graph, r: int, edge: Tuple[int, int], isom_inv: dict
) -> list[Tuple[int, int]]:
    """assume edge is in isom and isom_inv (cf. _get_unique_path_to_root); the
    returned cycle edges are sorted according to isom_inv labels"""
    assert isom_inv.get(edge) is not None
    u_path = _get_unique_path_to_root(tree, r, edge[0], isom_inv)
    v_path = _get_unique_path_to_root(tree, r, edge[1], isom_inv)
    cycle = set()
    duplicates = set()  # shared path to the root
    for e in u_path:
        cycle.add(e)
    for e in v_path:
        if e not in cycle:
            cycle.add(e)
        else:
            duplicates.add(e)
    for dup in duplicates:
        cycle.remove(dup)
    assert edge not in cycle
    cycle.add(edge)
    sorted_cycle = []
    for edge in cycle:
        order_label = isom_inv.get(edge)
        assert order_label is not None
        sorted_cycle.append((order_label, edge))
    sorted_cycle.sort(key=lambda x: x[0])
    return [edge for _, edge in sorted_cycle]


def get_cycles(
    roots: list[Graph], trees: list[Graph], isom_inv: dict
) -> list[Tuple[Tuple[int, int], list[Tuple[int, int]]]]:
    symmetry_cycles = []
    for root, tree in zip(roots, trees):
        r = root.vertices()[0]
        for edge in root.edges():
            if not tree.has_edge(edge[0], edge[1]):
                edge = (edge[0], edge[1])
                if isom_inv.get(edge) is None:
                    edge = (edge[1], edge[0])
                cycle = _get_symmetry_cycle(tree, r, edge, isom_inv)
                symmetry_cycles.append((edge, cycle))

    return symmetry_cycles
