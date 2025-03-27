"""Compute the currents in a graph using Kirchhoff's laws and nodal analysis."""
import os
import time

import networkx as nx
import numpy as np
from gauss_jordan import solve_system
from test_graphs import generate_test_graphs
from util import draw_circut, load_circuit, verify_circuit


def kirchhoff_analysis(graph, source, target, voltage):
    """Find the currents in the circuit using Kirchhoff's laws."""
    # Adding a direct edge between source and target with 0 resistance, representing the voltage source
    if graph.has_edge(source, target):
        graph[source][target]['resistance'] = 0
    elif graph.has_edge(target, source):
        graph[target][source]['resistance'] = 0
    else:
        graph.add_edge(source, target, resistance=0)

    edge_amount = graph.number_of_edges()
    edges = list(graph.edges())
    cycles = nx.cycle_basis(graph.to_undirected())

    matrix = np.zeros((edge_amount, edge_amount))
    vector = np.zeros(edge_amount)

    # First Kirchhoff's law - node analysis
    for i, node in enumerate(graph.nodes()):
        if len(cycles) + i >= edge_amount:   # No more nodes are processed when the system is fully defined
            break
        for (neighbor, node) in graph.in_edges(node):
            matrix[len(cycles) + i, edges.index((neighbor, node))] = 1
        for (node, neighbor) in graph.out_edges(node):
            matrix[len(cycles) + i, edges.index((node, neighbor))] = -1

    # Second Kirchhoff's law - mesh analysis
    for i, cycle in enumerate(cycles):
        for node_1, node_2 in zip(cycle, cycle[1:] + [cycle[0]]):   # cycle[1:] + [cycle[0]] - cycle is shifted by 1
            if (node_1, node_2) == (source, target):
                vector[i] = voltage
            elif (node_1, node_2) == (target, source):
                vector[i] = -voltage
            elif (node_1, node_2) in edges:
                matrix[i, edges.index((node_1, node_2))] = graph[node_1][node_2]['resistance']
            else:   #(node_1, node_2) in edges
                matrix[i, edges.index((node_2, node_1))] = -graph[node_2][node_1]['resistance']

    currents = solve_system(matrix, vector)

    # Adding currents to the graph and adjusting edge directions
    for i, (node_1, node_2) in enumerate(graph.copy().edges()):
        if currents[i] < 0:
            resistance = graph.edges[node_1, node_2]['resistance']
            graph.remove_edge(node_1, node_2)
            graph.add_edge(node_2, node_1, resistance=resistance)
            (node_1, node_2), currents[i] = (node_2, node_1), -currents[i]
        graph.edges[node_1, node_2]['current'] = currents[i]


def main():
    """Compute and visualize currents in a graph representing an electrical circuit."""
    DEFAULT_SOURCE = 0
    DEFAULT_TARGET = 1
    DEFAULT_VOLTAGE = 10
    DEFAULT_NODES = 50
    DEFAULT_EPS = 1e-10

    print("Enter parameters (press Enter to use default values):")
    start_input = input(f"Start node (default = {DEFAULT_SOURCE}): ")
    source = int(start_input) if start_input else DEFAULT_SOURCE
    target_input = input(f"Target node (default = {DEFAULT_TARGET}): ")
    target = int(target_input) if target_input else DEFAULT_TARGET
    voltage_input = input(f"Voltage (default = {DEFAULT_VOLTAGE}): ")
    voltage = int(voltage_input) if voltage_input else DEFAULT_VOLTAGE
    nodes_input = input(f"Node amount (default = {DEFAULT_NODES}): ")
    nodes = int(nodes_input) if nodes_input else DEFAULT_NODES
    while nodes % 2 != 0:
        nodes_input = input("NODE AMOUNT MUST BE EVEN: ")
        nodes = int(nodes_input) if nodes_input else DEFAULT_NODES
    eps_input = input(f"Epsilon (default = {DEFAULT_EPS}): ")
    eps = float(eps_input) if eps_input else DEFAULT_EPS

    generate_test_graphs(nodes)
    directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_graphs")
    graphs = {
        "Erdos-Renyi":      load_circuit(os.path.join(directory, "erdos_renyi.txt")),
        "Cubic":            load_circuit(os.path.join(directory, "cubic_graph.txt")),
        "Grid":             load_circuit(os.path.join(directory, "grid_graph.txt")),
        "Small world":      load_circuit(os.path.join(directory, "small_world_graph.txt")), 
        "Bridge-connected": load_circuit(os.path.join(directory, "bridge_connected_graph.txt"))
        }

    for graph_type, graph in graphs.items():
        start_time = time.time()
        kirchhoff_analysis(graph, source, target, voltage)
        end_time = time.time()
        print(f"{graph_type}: {"OK" if verify_circuit(graph, source, target, voltage, eps)\
                               else "FAILED"}")
        print(f"Time elapsed: {end_time-start_time:.2f}s")
        draw_circut(graph, graph_type, eps)


if __name__ == "__main__":
    main()
