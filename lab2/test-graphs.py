import networkx as nx
import numpy as np

def save_graph_to_file(G, filename):
    with open(filename, 'w') as f:
        for u, v in G.edges():
            resistance = np.random.uniform(0.5, 10)  # losowy opór
            f.write(f"{u} {v} {resistance}\n")

def generate_test_graphs():
    graph_types = {
        "random_graph.txt": nx.erdos_renyi_graph(50, 0.1),
        "regular_graph.txt": nx.random_regular_graph(3, 50),
        "two_random_graphs.txt": nx.disjoint_union(nx.erdos_renyi_graph(25, 0.2), nx.erdos_renyi_graph(25, 0.2)),
        "grid_graph.txt": nx.convert_node_labels_to_integers(nx.grid_2d_graph(7, 7)),
        "small_world_graph.txt": nx.watts_strogatz_graph(50, 4, 0.2)
    }
    
    for filename, G in graph_types.items():
        save_graph_to_file(G, filename)

generate_test_graphs()