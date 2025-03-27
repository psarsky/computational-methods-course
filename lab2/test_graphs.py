"""This module contains functions which generate random graphs with electrical resistances on the edges."""
import os

import networkx as nx
import numpy as np


def save_graph_to_file(graph, filename):
    """Save graph data to a text file."""
    if not os.path.exists("./test_graphs"):
        os.mkdir("./test_graphs")
    with open("test_graphs/" + filename, 'w', encoding="utf-8") as file:
        for node_1, node_2 in graph.edges():
            resistance = np.random.uniform(0.5, 10)
            file.write(f"{node_1} {node_2} {resistance}\n")
        if filename == "bridge_connected_graph.txt":
            resistance = np.random.uniform(0.5, 10)
            node = graph.number_of_nodes() // 2 + 1
            file.write(f"0 {node} {resistance}\n")


def generate_test_graphs(nodes):
    """Generate test graphs."""
    graph_types = {
        "erdos_renyi.txt": nx.erdos_renyi_graph(nodes, 0.1),
        "cubic_graph.txt": nx.random_regular_graph(3, nodes),
        "grid_graph.txt": nx.convert_node_labels_to_integers(nx.grid_2d_graph(int(np.sqrt(nodes)),
                                                                              int(np.sqrt(nodes)))),
        "small_world_graph.txt": nx.watts_strogatz_graph(nodes, 4, 0.2),
        "bridge_connected_graph.txt": nx.disjoint_union(nx.erdos_renyi_graph(nodes // 2, 0.2),
                                                        nx.erdos_renyi_graph(nodes // 2, 0.2))
    }

    for filename, graph in graph_types.items():
        save_graph_to_file(graph, filename)

if __name__ == "__main__":
    generate_test_graphs(50)
