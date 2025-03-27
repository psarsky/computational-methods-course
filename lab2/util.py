"""Utilities for loading, verifying, and visualizing circuits."""
import matplotlib.pyplot as plt
import networkx as nx


def load_circuit(filename):
    """Load circuit data from a text file."""
    graph = nx.DiGraph()
    with open(filename, 'r', encoding="utf-8") as file:
        for line in file:
            node_1, node_2, resistance = map(float, line.split())
            graph.add_edge(int(node_1), int(node_2), resistance=resistance)
    return graph


def verify_circuit(graph, source, target, voltage, eps):
    """Verify if the circuit satisfies Kirchhoff's laws."""

    current_max = max(current for _, _, current in graph.edges(data='current'))
    cycles = nx.cycle_basis(graph.to_undirected())

    # First Kirchhoff's law
    for node in graph.nodes():
        current = 0

        for edge in graph.in_edges(node):   # edges coming into the node
            current += graph.edges[edge]['current']
        for edge in graph.out_edges(node):  # edges coming out of the node
            current -= graph.edges[edge]['current']

        if current > eps * current_max:
            return False

    # Second Kirchhoff's law
    for cycle in cycles:
        cycle_voltage = 0
        for node_1, node_2 in zip(cycle, cycle[1:] + [cycle[0]]):   # cycle[1:] + [cycle[0]] - cycle is shifted by 1
            if (node_1, node_2) == (source, target):
                cycle_voltage += voltage
            elif (node_1, node_2) == (target, source):
                cycle_voltage -= voltage
            elif (node_1, node_2) in graph.edges():
                cycle_voltage -= graph.edges[node_1, node_2]['resistance'] * graph.edges[node_1, node_2]['current']
            else:   # (node_2, node_1) in graph.edges()
                cycle_voltage += graph.edges[node_2, node_1]['resistance'] * graph.edges[node_2, node_1]['current']

        if cycle_voltage > eps * voltage:
            return False

    return True


def draw_circut(graph, graph_type, eps):
    """Visualize the circuit with currents."""
    currents = [current for _, _, current in graph.edges(data='current')]
    current_min = min(currents)
    current_max = max(currents)
    edge_labels = {e: f"{(i if i > eps * current_max else 0):.3g} A"
                   for e, i in nx.get_edge_attributes(graph, 'current').items()}

    plt.figure(figsize=(16, 9))
    pos = nx.kamada_kawai_layout(graph)
    nx.draw_networkx_nodes(graph, pos, node_color='black')
    nx.draw_networkx_labels(graph, pos, font_color='white')
    nx.draw_networkx_edges(graph, pos, width=2, edge_color=currents, edge_cmap=plt.cm.plasma,
                           edge_vmin=current_min, edge_vmax=current_max, arrowsize=30)
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=7)
    plt.title(f"{graph_type} circuit")
    plt.axis('off')
    plt.show()
