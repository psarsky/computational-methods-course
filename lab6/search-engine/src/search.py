"""Module for searching documents based on a query."""

import re

import numpy as np


def query_to_vector(query_text, vocabulary, idf_values=None):
    """Transforms a text query into a bag-of-words feature vector.

    Args:
        query_text (str): Query text.
        vocabulary (dict): Dictionary mapping terms to their indices.
        idf_values (numpy.ndarray, optional): Array of IDF values for terms.

    Returns:
        numpy.ndarray: Query feature vector.
    """
    query_text = query_text.lower()
    terms = re.findall(r"\b[a-z]{3,}\b", query_text)

    query_vector = np.zeros(len(vocabulary))

    for term in terms:
        if term in vocabulary:
            term_idx = vocabulary[term]
            query_vector[term_idx] += 1

    if idf_values is not None:
        query_vector = query_vector * idf_values

    norm = np.linalg.norm(query_vector)
    if norm > 0:
        query_vector = query_vector / norm

    return query_vector


def search_documents(
    query_text, normalized_matrix, vocabulary, doc_ids, idf_values, k=10
):
    """Searches for documents most similar to the query.

    Args:
        query_text (str): Query text.
        normalized_matrix (scipy.sparse.csr_matrix): Normalized term-document matrix.
        vocabulary (dict): Dictionary mapping terms to their indices.
        doc_ids (list): List of document identifiers.
        idf_values (numpy.ndarray, optional): Array of IDF values for terms.
        k (int, optional): Number of results to return. Defaults to 10.

    Returns:
        list: List of tuples (doc_id, similarity_score).
    """
    query_vector = query_to_vector(query_text, vocabulary, idf_values)

    similarities = np.abs(query_vector @ normalized_matrix)

    top_indices = np.argsort(-similarities)[:k]

    results = [(doc_ids[i], similarities[i]) for i in top_indices]

    return results


def search_documents_svd(
    query_text, svd_components, vocabulary, doc_ids, idf_values, k=10
):
    """Searches for documents most similar to the query using SVD to reduce noise.

    Args:
        query_text (str): Query text.
        svd_components (tuple): A tuple containing:
            - U_k (numpy.ndarray): Matrix of left singular vectors.
            - D_k (numpy.ndarray): Array of singular values.
            - V_T_k (numpy.ndarray): Matrix of right singular vectors.
        vocabulary (dict): Dictionary mapping terms to their indices.
        doc_ids (list): List of document identifiers.
        idf_values (numpy.ndarray, optional): Array of IDF values for terms.
        k (int, optional): Number of results to return. Defaults to 10.

    Returns:
        list: List of tuples (doc_id, similarity_score).
    """
    U_k, D_k, V_T_k = svd_components

    query_vector = query_to_vector(query_text, vocabulary, idf_values)

    query_concepts = query_vector @ U_k @ np.diag(1.0 / D_k)

    doc_concepts = V_T_k.T

    doc_norms = np.sqrt(np.sum(doc_concepts**2, axis=1))
    doc_norms = np.maximum(doc_norms, 1e-10)
    normalized_docs = doc_concepts / doc_norms[:, np.newaxis]

    query_norm = np.linalg.norm(query_concepts)
    if query_norm > 0:
        query_concepts = query_concepts / query_norm

    similarities = np.abs(query_concepts @ normalized_docs.T)

    top_indices = np.argsort(-similarities)[:k]
    results = [(doc_ids[idx], similarities[idx]) for idx in top_indices]

    return results
