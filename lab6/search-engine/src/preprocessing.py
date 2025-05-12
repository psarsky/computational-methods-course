"""Module for pre-processing documents and creating a term-document matrix."""

import os
import re
import sqlite3
from collections import Counter

# import nltk
import numpy as np
from nltk.corpus import stopwords
from scipy.sparse import csr_matrix, diags


def build_vocabulary(db_path, min_df=5, max_df=0.5, max_features=100000):
    """Builds a term dictionary from documents in the database.

    Args:
        db_path (string): Path to the SQLite database.
        min_df (int, optional): Minimum number of documents a term must appear in. Defaults to 5.
        max_df (float, optional): Maximum percentage of documents a term can appear in. Defaults to 0.5.
        max_features (int, optional): Maximum number of terms in the dictionary. Defaults to 10000.

    Returns:
        dict: A dictionary mapping terms to their indices.
    """
    # nltk.download("stopwords")
    stop_words = set(stopwords.words("english"))

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM pages")
    n_documents = cursor.fetchone()[0]

    term_doc_counts = Counter()
    all_terms = set()

    cursor.execute("SELECT id, content FROM pages")
    for _, content in cursor.fetchall():
        content = content.lower()
        terms = re.findall(r"\b[a-z]{3,}\b", content)

        terms = [term for term in terms if term not in stop_words]

        unique_terms = set(terms)
        all_terms.update(unique_terms)
        for term in unique_terms:
            term_doc_counts[term] += 1

    filtered_terms = [
        term
        for term, count in term_doc_counts.items()
        if min_df <= count <= max_df * n_documents
    ]

    if len(filtered_terms) > max_features:
        sorted_terms = sorted(filtered_terms, key=lambda t: -term_doc_counts[t])
        final_terms = sorted_terms[:max_features]
    else:
        final_terms = filtered_terms

    vocabulary = {term: idx for idx, term in enumerate(final_terms)}

    conn.close()

    return vocabulary


def create_document_vectors(db_path, vocabulary):
    """Creates bag-of-words vectors for all documents.

    Args:
        db_path (string): Path to the SQLite database.
        vocabulary (dict): Dictionary mapping terms to their indices.

    Returns:
        tuple: A tuple containing:
            - doc_vectors (scipy.sparse.csr_matrix): Sparse matrix of document vectors.
            - doc_ids (list): List of document IDs.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM pages")
    n_documents = cursor.fetchone()[0]

    vocab_size = len(vocabulary)

    row_indices = []
    col_indices = []
    values = []
    doc_ids = []

    cursor.execute("SELECT id, content FROM pages")
    for doc_idx, (doc_id, content) in enumerate(cursor.fetchall()):
        doc_ids.append(doc_id)

        content = content.lower()
        terms = re.findall(r"\b[a-z]{3,}\b", content)

        term_counts = {}
        for term in terms:
            if term in vocabulary:
                term_idx = vocabulary[term]
                term_counts[term_idx] = term_counts.get(term_idx, 0) + 1

        for term_idx, count in term_counts.items():
            row_indices.append(term_idx)
            col_indices.append(doc_idx)
            values.append(count)

    term_doc_matrix = csr_matrix(
        (values, (row_indices, col_indices)), shape=(vocab_size, n_documents)
    )

    conn.close()

    return term_doc_matrix, doc_ids


def apply_idf(term_doc_matrix):
    """Applies IDF transformation to the term-document matrix.

    Args:
        term_doc_matrix (scipy.sparse.csr_matrix): Sparse matrix of term-document counts.

    Returns:
        tuple: A tuple containing:
            - tfidf_matrix (scipy.sparse.csr_matrix): Sparse matrix of TF-IDF values.
            - idf_values (numpy.ndarray): Array of IDF values for each term.
    """
    _, n_docs = term_doc_matrix.shape

    doc_counts = np.array((term_doc_matrix > 0).sum(axis=1)).flatten()

    idf_values = np.log(n_docs / (doc_counts + 1))  # +1 for stability

    idf_diag = diags(idf_values)
    tfidf_matrix = idf_diag.dot(term_doc_matrix)

    return tfidf_matrix, idf_values


if __name__ == "__main__":
    # for testing purposes
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data",
        "simplified_wiki_index.db",
    )
    voc = build_vocabulary(path)
    print("Vocabulary size:", len(voc))
    print("Sample vocabulary:", list(voc.items())[:100])
