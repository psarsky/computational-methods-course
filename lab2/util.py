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

    current_max = max(abs(current) for _, _, current in graph.edges(data='current'))
    cycles = nx.cycle_basis(graph.to_undirected())

    # KCL
    for node in graph.nodes():
        current_in = sum(graph.edges[edge]['current'] for edge in graph.in_edges(node))
        current_out = sum(graph.edges[edge]['current'] for edge in graph.out_edges(node))

        if abs(current_in - current_out) > eps * current_max:
            print(f"Node {node} violates KCL: In={current_in}, Out={current_out}")
            return False

    # KVL
    for cycle in cycles:
        cycle_voltage = 0

        for i, node in enumerate(cycle):
            neighbor = cycle[(i + 1) % len(cycle)]

            # Voltage source contribution
            if (node, neighbor) == (source, target):
                cycle_voltage += voltage
            elif (node, neighbor) == (target, source):
                cycle_voltage -= voltage

            # Resistor voltage drops
            if graph.has_edge(node, neighbor):
                cycle_voltage -= graph[node][neighbor]['resistance'] * graph[node][neighbor]['current']
            elif graph.has_edge(neighbor, node):
                cycle_voltage += graph[neighbor][node]['resistance'] * graph[neighbor][node]['current']

        if abs(cycle_voltage) > eps * voltage:
            print(f"KVL Violation in Cycle: {cycle}")
            return False

    return True


def draw_circut(graph, graph_type, eps):
    """Visualize the circuit with currents."""
    currents = [current for _, _, current in graph.edges(data='current')]
    current_min = 0
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
