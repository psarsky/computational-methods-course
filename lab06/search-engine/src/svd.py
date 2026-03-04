"""Module for applying SVD and low-rank approximation to a term-document matrix."""

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
