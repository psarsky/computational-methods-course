"""Utility functions for the PageRank algorithm."""

import gzip
import os
import shutil
import urllib.request

import numpy as np
from scipy.sparse import csr_matrix


def create_adj_matrix(G):
    """Creates a regular adjacency matrix from a graph representation."""
    n = G.number_of_nodes()
    A = np.zeros((n, n))

    for u in G.nodes():
        out_degree = G.out_degree(u)
        if out_degree > 0:
            for v in G.successors(u):
                A[v, u] = 1.0 / out_degree

    return A


def create_adj_matrix_sparse(G):
    """Creates a sparse adjacency matrix from a graph representation."""
    n = G.number_of_nodes()
    node_to_index = {node: i for i, node in enumerate(G.nodes())}
    rows, cols, data = [], [], []

    for u in G.nodes():
        out_degree = G.out_degree(u)
        if out_degree > 0:
            for v in G.successors(u):
                rows.append(node_to_index[v])
                cols.append(node_to_index[u])
                data.append(1.0 / out_degree)

    return csr_matrix((data, (rows, cols)), shape=(n, n))


def download_and_extract(url, output_file):
    """Downloads and extracts a gzipped file from a URL if it doesn't exist locally."""
    if not os.path.exists(output_file):
        gz_file = f"{output_file}.gz"
        if not os.path.exists(gz_file):
            print(f"Downloading {url}...")
            urllib.request.urlretrieve(url, gz_file)

        print(f"Extracting {gz_file}...")
        with gzip.open(gz_file, "rb") as f_in:
            with open(output_file, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

    return output_file
