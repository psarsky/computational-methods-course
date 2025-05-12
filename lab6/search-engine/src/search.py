"""Module for searching documents."""

import re

import numpy as np


def create_inverse_vocabulary(vocabulary):
    """Creates an inverse vocabulary.

    Args:
        vocabulary (dict): Dictionary mapping terms to their indices.

    Returns:
        dict: Dictionary mapping indices to terms.
    """
    return {idx: term for term, idx in vocabulary.items()}


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
    query_text, normalized_matrix, vocabulary, doc_ids, idf_values=None, k=10
):
    """Searches for documents most similar to the query.

    Args:
        query_text (str): Query text.
        term_doc_matrix (scipy.sparse.csr_matrix): Term-document matrix.
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

    results = [(doc_ids[idx], similarities[idx]) for idx in top_indices]

    return results
