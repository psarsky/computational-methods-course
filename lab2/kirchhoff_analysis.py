"""Compute the currents in a graph using Kirchhoff's laws and nodal analysis."""
import time

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from gauss_jordan import solve_system
from test_graphs import generate_test_graphs


def kirchhoff_analysis(graph, start, target, voltage):
    """Find the currents in the circuit using Kirchhoff's laws."""
    if graph.has_edge(start, target):
        graph[start][target]['resistance'] = 0
    elif graph.has_edge(target, start):
        graph[target][start]['resistance'] = 0
    else:
        graph.add_edge(start, target, resistance=0)

    edge_amount = graph.number_of_edges()
    matrix = np.zeros((edge_amount, edge_amount))
    vector = np.zeros(edge_amount)
    edge_list = list(graph.edges())

    cycles = nx.cycle_basis(graph.to_undirected())

    for i, node in enumerate(graph.nodes()):
        if len(cycles) + i == len(edge_list):
            break
        for (node_1, node) in graph.in_edges(node):
            j = edge_list.index((node_1, node))
            matrix[len(cycles) + i, j] = 1
        for (node, node_2) in graph.out_edges(node):
            j = edge_list.index((node, node_2))
            matrix[len(cycles) + i, j] = -1

    for i, cycle in enumerate(cycles):
        for pair in zip(cycle, cycle[1:] + [cycle[0]]):
            if pair == (start, target):
                vector[i] = voltage
            elif pair == (target, start):
                vector[i] = -voltage
            else:
                (node_1, node_2) = pair
                if (node_1, node_2) in edge_list:
                    j = edge_list.index((node_1, node_2))
                    matrix[i][j] = graph[node_1][node_2]['resistance']
                else:
                    j = edge_list.index((node_2, node_1))
                    matrix[i][j] = -graph[node_2][node_1]['resistance']

    current = solve_system(matrix, vector)

    for i, (node_1, node_2) in enumerate(graph.copy().edges()):
        if current[i] < 0:
            resistance = graph.edges[node_1, node_2]['resistance']
            graph.remove_edge(node_1, node_2)
            graph.add_edge(node_2, node_1, resistance=resistance)
            (node_1, node_2), current[i] = (node_2, node_1), -current[i]
        graph.edges[node_1, node_2]['current'] = current[i]

    return graph, cycles


def load_graph(filename):
    """Load graph data from a text file."""
    graph = nx.DiGraph()
    with open(filename, 'r', encoding="utf-8") as file:
        for line in file:
            node_1, node_2, resistance = map(float, line.split())
            graph.add_edge(int(node_1), int(node_2), resistance=resistance)
    return graph


def verify_circuit(graph, start, target, voltage, cycles, eps):
    """Verify if the circuit satisfies Kirchhoff's laws."""

    currents = [current for _, _, current in graph.edges(data='current')]
    current_max = max(currents)

    # First Kirchhoff's law
    for node in graph.nodes():
        current = 0
        for edge in graph.in_edges(node):
            current += graph.edges[edge]['current']
        for edge in graph.out_edges(node):
            current -= graph.edges[edge]['current']
        if current > eps * current_max:
            return False

    # Second Kirchhoff's law
    for cycle in cycles:
        cycle_voltage = 0
        for pair in zip(cycle, cycle[1:] + [cycle[0]]):
            if pair == (start, target):
                cycle_voltage += voltage
            elif pair == (target, start):
                cycle_voltage -= voltage
            else:
                (node_1, node_2) = pair
                if (node_1, node_2) in graph.edges():
                    cycle_voltage -= graph.edges[node_1, node_2]['resistance'] * graph.edges[node_1, node_2]['current']
                else: # (node_2, node_1) in graph.edges()
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

    pos = nx.kamada_kawai_layout(graph)
    nx.draw_networkx_nodes(graph, pos, node_color='black')
    nx.draw_networkx_labels(graph, pos, font_color='white')
    nx.draw_networkx_edges(graph, pos, width=2, edge_color=currents, edge_cmap=plt.cm.plasma,
                           edge_vmin=current_min, edge_vmax=current_max, arrowsize=30)
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=7)
    plt.title(f"{graph_type} circuit")
    plt.axis('off')
    plt.show()


def main():
    """Compute and visualize currents in a graph representing an electrical circuit."""
    print("Enter parameters (press Enter to use default values):")
    start_input = input("Start node (default = 0): ")
    start = int(start_input) if start_input else 0
    target_input = input("Target node (default = 1): ")
    target = int(target_input) if target_input else 1
    voltage_input = input("Voltage (default = 10): ")
    voltage = int(voltage_input) if voltage_input else 10
    nodes_input = input("Node amount (default = 50): ")
    nodes = int(nodes_input) if nodes_input else 50
    while nodes % 2 != 0:
        nodes_input = input("NODE AMOUNT MUST BE EVEN: ")
        nodes = int(nodes_input) if nodes_input else 50
    eps_input = input("Epsilon (default = 1e-10): ")
    eps = float(eps_input) if eps_input else 1e-10

    generate_test_graphs(nodes)

    graphs = {"Erdos-Renyi":      load_graph("test_graphs/erdos_renyi.txt"),
              "Cubic":            load_graph("test_graphs/cubic_graph.txt"),
              "Grid":             load_graph("test_graphs/grid_graph.txt"),
              "Small world":      load_graph("test_graphs/small_world_graph.txt"), 
              "Bridge-connected": load_graph("test_graphs/bridge_connected_graph.txt")}

    for graph_type, graph in graphs.items():
        start_time = time.time()
        graph, cycles = kirchhoff_analysis(graph, start, target, voltage)
        end_time = time.time()
        print(f"{graph_type}: {"OK" if verify_circuit(graph, start, target, voltage, cycles, eps)\
                               else "FAILED"}")
        print(f"Time elapsed: {end_time-start_time:.2f}s")
        draw_circut(graph, graph_type, eps)


if __name__ == "__main__":
    main()
