"""Utility functions for the random walk ranking algorithm."""

import numpy as np


def generate_random_directed_graph(n, p=0.3, seed=None):
    """Creates a random directed graph with n nodes and edge probability p."""
    if seed is not None:
        np.random.seed(seed)

    graph = [[] for _ in range(n)]

    for u in range(n):
        for v in range(n):
            if u != v and np.random.random() < p:
                graph[u].append(v)

    return graph


def create_adj_matrix(graph):
    """Creates an adjacency matrix from a graph representation."""
    n = len(graph)
    A = np.zeros((n, n))

    for u in range(n):
        outgoing_edges = graph[u]
        Nu = len(outgoing_edges)
        if Nu > 0:
            for v in outgoing_edges:
                A[v, u] = 1 / Nu

    return A
