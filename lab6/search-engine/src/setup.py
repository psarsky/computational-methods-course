"""Module for setting up the search engine by loading data and configuring parameters."""

import os
import pickle
import time

from scipy.sparse import load_npz, save_npz
from sklearn.preprocessing import normalize
from src.preprocessing import (apply_idf, build_vocabulary,
                               create_document_vectors)
from src.svd import apply_svd

DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "simplified_wiki_index.db",
)


def setup():
    """Setup function for the search engine.

    Returns:
        tuple: A tuple containing:
            - use_svd (bool): Whether to use SVD.
            - svd_components (tuple): SVD components if SVD is used.
            - vocabulary (dict): Dictionary mapping terms to their indices.
            - doc_ids (list): List of document identifiers.
            - idf_values (numpy.ndarray): Array of IDF values for terms.
            - k_results (int): Number of results to return.
    """
    use_idf_input = input("Use IDF (y/n, default: y): ").lower()
    use_idf = not use_idf_input.startswith("n")

    use_svd_input = input("Use SVD (y/n, default: y): ").lower()
    use_svd = not use_svd_input.startswith("n")

    if use_svd:
        k_svd_input = input("Number of singular values (default: 100): ")
        k_svd = int(k_svd_input) if k_svd_input.isdigit() else 100

    k_results_input = input("Amount of results to return (default: 10): ")
    k_results = int(k_results_input) if k_results_input.isdigit() else 10

    setup_start = time.time()
    vocabulary = load_or_compute_vocabulary()

    term_doc_matrix, doc_ids = load_or_compute_term_doc_matrix(vocabulary)

    idf_values = None
    if use_idf:
        print("\nApplying IDF transformation...")
        idf_start = time.time()
        term_doc_matrix, idf_values = apply_idf(term_doc_matrix)
        idf_time = time.time() - idf_start
        print(f"TIME: {idf_time:.2f}s")

    print("\nNormalizing term-document matrix...")
    norm_start = time.time()
    normalized_matrix = normalize(term_doc_matrix, axis=1)
    norm_time = time.time() - norm_start
    print(f"TIME: {norm_time:.2f}s")

    svd_components = None
    if use_svd:
        svd_components = load_or_compute_svd(use_idf, normalized_matrix, k_svd)

    setup_time = time.time() - setup_start

    print(f"Total setup time: {setup_time:.2f}s\n")

    return (
        use_svd,
        normalized_matrix,
        svd_components,
        vocabulary,
        doc_ids,
        idf_values,
        k_results,
    )


def load_or_compute_vocabulary():
    """Loads precomputed vocabulary from file or builds it if file doesn't exist.

    Returns:
        dict: Dictionary mapping terms to their indices.
    """
    vocab_file_name = "vocabulary.pkl"
    data_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
    )
    os.makedirs(data_dir, exist_ok=True)
    vocab_file_path = os.path.join(data_dir, vocab_file_name)

    if os.path.exists(vocab_file_path):
        print("\nLoading precomputed vocabulary from file...")
        try:
            with open(vocab_file_path, "rb") as vocab_file:
                vocabulary = pickle.load(vocab_file)
            return vocabulary
        except Exception as e:
            print(f"Error loading vocabulary: {str(e)}")

    print("\nBuilding vocabulary...")
    start_time = time.time()
    vocabulary = build_vocabulary(DB_PATH)
    build_time = time.time() - start_time
    print(f"TIME: {build_time:.2f}s, SIZE: {len(vocabulary)}")

    print("Saving vocabulary...")
    try:
        with open(vocab_file_path, "wb") as vocab_file:
            pickle.dump(vocabulary, vocab_file)
        print("Vocabulary saved successfully.")
    except Exception as e:
        print(f"Error saving vocabulary: {str(e)}")

    return vocabulary


def load_or_compute_term_doc_matrix(vocabulary):
    """Loads precomputed term-document matrix from file or computes it if file doesn't exist.

    Args:
        vocabulary (dict): Dictionary mapping terms to their indices.

    Returns:
        tuple: A tuple containing:
            - term_doc_matrix (scipy.sparse.csr_matrix): Sparse matrix of document vectors.
            - doc_ids (list): List of document IDs.
    """
    term_doc_file_name = "term_doc_matrix.npz"
    doc_ids_file_name = "doc_ids.pkl"

    data_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
    )
    os.makedirs(data_dir, exist_ok=True)

    term_doc_file_path = os.path.join(data_dir, term_doc_file_name)
    doc_ids_file_path = os.path.join(data_dir, doc_ids_file_name)

    if os.path.exists(term_doc_file_path) and os.path.exists(doc_ids_file_path):
        print(
            "\nLoading precomputed term-document matrix and document IDs from file..."
        )
        try:
            term_doc_matrix = load_npz(term_doc_file_path)
            with open(doc_ids_file_path, "rb") as ids_file:
                doc_ids = pickle.load(ids_file)
            return term_doc_matrix, doc_ids
        except Exception as e:
            print(f"Error loading term-document matrix: {str(e)}")

    print("\nCreating term-document matrix...")
    start_time = time.time()
    term_doc_matrix, doc_ids = create_document_vectors(DB_PATH, vocabulary)
    build_time = time.time() - start_time
    print(
        f"TIME: {build_time:.2f}s, SIZE: {term_doc_matrix.shape[0]} x {term_doc_matrix.shape[1]}"
    )

    print("Saving term-document matrix and document IDs...")
    try:
        save_npz(term_doc_file_path, term_doc_matrix)
        with open(doc_ids_file_path, "wb") as ids_file:
            pickle.dump(doc_ids, ids_file)
        print("Term-document matrix and document IDs saved successfully.")
    except Exception as e:
        print(f"Error saving term-document matrix: {str(e)}")

    return term_doc_matrix, doc_ids


def load_or_compute_svd(use_idf, normalized_matrix, k_svd):
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
    data_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data",
        f"svd-components-{"idf" if use_idf else "no-idf"}",
    )
    os.makedirs(data_dir, exist_ok=True)
    svd_file_path = os.path.join(data_dir, svd_file_name)

    if os.path.exists(svd_file_path):
        print(f"\nLoading precomputed SVD components (k={k_svd}) from file...")
        try:
            with open(svd_file_path, "rb") as svd_file:
                svd_components = pickle.load(svd_file)
            return svd_components
        except Exception as e:
            print(f"Error loading SVD components: {str(e)}")

    print(f"\nApplying SVD (k={k_svd})...")
    start_time = time.time()
    svd_components = apply_svd(normalized_matrix, k_svd)
    build_time = time.time() - start_time
    print(f"TIME: {build_time:.2f}s\n")

    print("Saving SVD components...")
    try:
        with open(svd_file_path, "wb") as svd_file:
            pickle.dump(svd_components, svd_file)
        print("SVD components saved successfully.")
    except Exception as e:
        print(f"Error saving SVD components: {str(e)}")

    return svd_components
