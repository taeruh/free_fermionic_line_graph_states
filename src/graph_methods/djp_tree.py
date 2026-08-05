# Finding the minimum spanning tree using Dijkstra-Jarník-Prim algorithm.
#
# We actually want to find the maximum spanning tree, but this can be done exactly the
# same way by replacing/switching the terms "less"<->"more" and "min"<->"max" in the
# algorithm and proof (alternatively we could just negate all weights). At the end of this
# file is the proof copied from Wikipedia with the necessary changes.
#
# EDIT: maybe we should just use sage.graphs.base.boost_graph.min_spanning_tree

from sage.all import Graph
import numpy as np


def maximum_spanning_tree(g: Graph) -> Graph:
    num_vertices = g.num_verts()
    vertices = [next(g.vertex_iterator())]
    edges = []

    while len(vertices) < num_vertices:
        max_edge, max_weight, added_vertex = None, float("-inf"), None
        for v in vertices:
            for u in g.neighbors(v):
                if u not in vertices:
                    weight = np.abs(g.edge_label(v, u))
                    if weight > max_weight:
                        max_edge, max_weight = (v, u, weight), weight
                        added_vertex = u
        edges.append(max_edge)
        vertices.append(added_vertex)

    ret = Graph()
    for edge in edges:
        ret.add_edge(edge)
    return ret


# Let P be a connected, weighted graph. At every iteration of Prim's algorithm, an edge
# must be found that connects a vertex in a subgraph to a vertex outside the subgraph.
# Since P is connected, there will always be a path to every vertex. The output Y of
# Prim's algorithm is a tree, because the edge and vertex added to tree Y are connected.

# Let Y1 be a maximum spanning tree of graph P. If Y1=Y then Y is a maximum spanning tree.
# Otherwise, let e be the first edge added during the construction of tree Y that is not
# in tree Y1, and V be the set of vertices connected by the edges added before edge e.
# Then one endpoint of edge e is in set V and the other is not. Since tree Y1 is a
# spanning tree of graph P, there is a path in tree Y1 joining the two endpoints. As one
# travels along the path, one must encounter an edge f joining a vertex in set V to one
# that is not in set V. Now, at the iteration when edge e was added to tree Y, edge f
# could also have been added and it would be added instead of edge e if its weight was
# more than e, and since edge f was not added, we conclude that w(e) geq w(f). Let tree Y2
# be the graph obtained by removing edge f from and adding edge e to tree Y1. It is easy
# to show that tree Y2 is connected, has the same number of edges as tree Y1, and the
# total weights of its edges is not smaller than that of tree Y1, therefore it is also a
# maximum spanning tree of graph P and it contains edge e and all the edges added before
# it during the construction of set V. Repeat the steps above and we will eventually
# obtain a maximum spanning tree of graph P that is identical to tree Y. This shows Y is a
# maximum spanning tree. The maximum spanning tree allows for the first subset of the
# sub-region to be expanded into a larger subset X, which we assume to be the minimum.
