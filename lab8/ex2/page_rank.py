import os
import time

import networkx as nx
import numpy as np
from util import (create_adj_matrix, create_adj_matrix_sparse,
                  download_and_extract)
from vis import (display_results, plot_pagerank_histogram,
                 plot_pagerank_histogram_large, visualize_pagerank_comparison)


def pagerank_power_method(A, e=None, d=0.85, max_iter=100, tol=1e-6):
    n = A.shape[0]

    if e is None:
        e = np.ones(n) / n

    r = np.ones(n) / n

    for i in range(max_iter):
        r_next = d * A @ r

        delta_norm = np.linalg.norm(r, 1) - np.linalg.norm(r_next, 1)
        r_next = r_next + delta_norm * e

        delta = np.linalg.norm(r_next - r, 1)

        if delta < tol:
            return r_next, i + 1

        r = r_next

    return r, max_iter


def test_karate_club(damping_factors):
    G = nx.karate_club_graph()
    G = G.to_directed()

    print(
        f"Karate Club graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges"
    )

    n = G.number_of_nodes()

    jump_vectors = {
        "uniform": np.ones(n) / n,
        "first_biased": np.array([0.5] + [0.5 / (n - 1)] * (n - 1)),
        "half_biased": np.array(
            [0.7 / (n // 2)] * (n // 2) + [0.3 / (n - n // 2)] * (n - n // 2)
        ),
    }

    A = create_adj_matrix(G)
    results = test_jump_vectors(A, damping_factors, jump_vectors)

    print("\nResults for Karate Club graph:")
    display_results(results)

    return G, results


def test_epinions(url, file_path, damping_factors):
    file_path = download_and_extract(url, file_path)

    print(f"Loading graph from {file_path}...")
    G = nx.read_edgelist(file_path, create_using=nx.DiGraph(), comments="#")

    print(f"Epinions graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    n = G.number_of_nodes()

    jump_vectors = {
        "uniform": np.ones(n) / n,
        "first_biased": np.array([0.5] + [0.5 / (n - 1)] * (n - 1)),
        "half_biased": np.array(
            [0.7 / (n // 2)] * (n // 2) + [0.3 / (n - n // 2)] * (n - n // 2)
        ),
    }

    A_sparse = create_adj_matrix_sparse(G)
    results = test_jump_vectors(A_sparse, damping_factors, jump_vectors)

    print("\nResults for Epinions graph:")
    display_results(results)

    return results


def test_jump_vectors(A, damping_factors, jump_vectors):
    results = {}

    for d in damping_factors:
        d_results = {}
        for name, e in jump_vectors.items():
            start_time = time.time()
            pr, iterations = pagerank_power_method(A, e=e, d=d)
            end_time = time.time()

            d_results[name] = {
                "pr": pr,
                "iterations": iterations,
                "time": end_time - start_time,
                "top5": np.argsort(-pr)[:5],
            }
        results[d] = d_results

    return results


def main():
    damping_factors = [0.9, 0.85, 0.75, 0.6, 0.5]

    print("Testing PageRank with jumps on Karate Club graph\n")
    graph, results_karate = test_karate_club(damping_factors)

    visualize_pagerank_comparison(graph, results_karate, 0.5)

    for name in results_karate[0.5]:
        pr = results_karate[0.5][name]["pr"]
        plot_pagerank_histogram(pr, f"Karate Club - {name} (d=0.5)")

    print("\nTesting PageRank with jumps on Epinions graph (~75k nodes)\n")

    large_graph_url = "https://snap.stanford.edu/data/soc-Epinions1.txt.gz"
    directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_graphs")
    os.makedirs(directory, exist_ok=True)
    large_graph_file = os.path.join(directory, "soc-Epinions1.txt")

    results_large = test_epinions(large_graph_url, large_graph_file, damping_factors)

    for name in results_large[0.5]:
        pr = results_large[0.5][name]["pr"]
        plot_pagerank_histogram_large(pr, f"Epinions - {name} (d=0.5)")


if __name__ == "__main__":
    main()
