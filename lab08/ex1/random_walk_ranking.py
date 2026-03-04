"""Random walk ranking algorithm for directed graphs."""

import time

import networkx as nx
import numpy as np
from util import create_adj_matrix, generate_random_directed_graph
from vis import display_results, visualize_results


def random_walk(A, d=0.85, max_iter=100, tol=1e-6):
    """Implements the random walk algorithm for node ranking."""
    n = A.shape[0]
    r = np.ones(n) / n

    for i in range(max_iter):
        r_next = d * (A @ r) + (1 - d) * np.ones(n) / n
        if np.linalg.norm(r_next - r, 1) < tol:
            return r_next, i + 1
        r = r_next

    return r, max_iter


def test_all(graph, d=0.85):
    """Tests the random walk algorithm on a given graph."""
    adjacency_matrix = create_adj_matrix(graph)

    start_time = time.time()
    ranking_scores, iterations = random_walk(adjacency_matrix, d=d)
    computation_time = time.time() - start_time

    G = nx.DiGraph()
    for node, neighbors in enumerate(graph):
        G.add_node(node)
        for neighbor in neighbors:
            G.add_edge(node, neighbor)

    top5_indices = np.argsort(ranking_scores)[-5:][::-1]

    result = {
        "pr": ranking_scores,
        "iterations": iterations,
        "time": computation_time,
        "top5": top5_indices,
        "G": G,
    }

    return result


def main():
    """Main function that creates test graphs and runs the random walk algorithm."""
    graphs = {
        "15-node graph": generate_random_directed_graph(15, p=0.2, seed=123),
        "25-node graph": generate_random_directed_graph(25, p=0.15, seed=456),
        "40-node graph": generate_random_directed_graph(40, p=0.1, seed=789),
    }

    damping_factors = [0.85]

    results = {}

    for graph_name, graph in graphs.items():
        results[graph_name] = {}

        for d in damping_factors:
            result = test_all(graph, d)

            edge_count = sum(len(neighbors) for neighbors in graph)

            results[graph_name][d] = result
            results[graph_name][d]["vertices"] = len(graph)
            results[graph_name][d]["edges"] = edge_count

    display_results(results)

    visualize_results(results)


if __name__ == "__main__":
    main()
