"""Module for applying SVD and low-rank approximation to a term-document matrix."""

import os
import pickle
import time

import numpy as np
from scipy.sparse.linalg import svds


def apply_svd(term_doc_matrix, k):
    """Applies singular value decomposition (SVD) to a term-document matrix
    and returns a low-rank approximation of the matrix.

    Args:
        term_doc_matrix (csr_matrix): Term-document matrix.
        k (int): Number of singular values to keep.

    Returns:
        tuple: A tuple containing:
            - U_k (numpy.ndarray): Matrix of left singular vectors.
            - D_k (numpy.ndarray): Diagonal matrix of singular values.
            - V_T_k (numpy.ndarray): Matrix of right singular vectors.
    """
    U, D, V_T = svds(term_doc_matrix.tocsc(), k=k)

    idx = np.argsort(-D)
    D_k = D[idx]
    U_k = U[:, idx]
    V_T_k = V_T[idx, :]

    return U_k, D_k, V_T_k


def load_or_compute_svd(normalized_matrix, k_svd):
    """Loads precomputed SVD components from file or computes them if file doesn't exist.

    Args:
        normalized_matrix (scipy.sparse.csr_matrix): Normalized term-document matrix.
        k_svd (int): Number of singular values.

    Returns:
        tuple: A tuple containing:
            - U_k (numpy.ndarray): Matrix of left singular vectors.
            - D_k (numpy.ndarray): Diagonal matrix of singular values.
            - V_T_k (numpy.ndarray): Matrix of right singular vectors.
    """
    svd_file_name = f"svd_components_k{k_svd}.pkl"
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    os.makedirs(data_dir, exist_ok=True)
    svd_file_path = os.path.join(data_dir, svd_file_name)

    if os.path.exists(svd_file_path):
        print(f"Loading precomputed SVD components (k={k_svd}) from file...")
        load_start = time.time()
        try:
            with open(svd_file_path, "rb") as svd_file:
                svd_components = pickle.load(svd_file)
            load_time = time.time() - load_start
            print(f"TIME: {load_time:.2f}s\n")
            return svd_components
        except Exception as e:
            print(f"Error loading SVD components: {str(e)}")

    print(f"Applying SVD (k={k_svd})...")
    svd_start = time.time()
    svd_components = apply_svd(normalized_matrix, k_svd)
    svd_time = time.time() - svd_start
    print(f"TIME: {svd_time:.2f}s\n")

    print("Saving SVD components...")
    try:
        with open(svd_file_path, "wb") as svd_file:
            pickle.dump(svd_components, svd_file)
        print("SVD components saved successfully.\n")
    except Exception as e:
        print(f"Error saving SVD components: {str(e)}\n")

    return svd_components
