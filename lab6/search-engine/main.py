"""Main module for the search engine."""

import os
import sqlite3
import time

from sklearn.preprocessing import normalize
from src.preprocessing import (apply_idf, build_vocabulary,
                               create_document_vectors)
from src.search import search_documents, search_documents_svd
from src.svd import load_or_compute_svd

DB_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
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

    print()

    print("Building term dictionary...")
    dict_start = time.time()
    vocabulary = build_vocabulary(DB_PATH)
    dict_time = time.time() - dict_start
    print(f"TIME: {dict_time:.2f}s, SIZE: {len(vocabulary)}\n")

    print("Creating term-document matrix...")
    m_start = time.time()
    term_doc_matrix, doc_ids = create_document_vectors(DB_PATH, vocabulary)
    m_time = time.time() - m_start
    print(
        f"TIME: {m_time:.2f}s, SIZE: {term_doc_matrix.shape[0]} x {term_doc_matrix.shape[1]}\n"
    )

    idf_values = None
    if use_idf:
        print("Applying IDF transformation...")
        idf_start = time.time()
        term_doc_matrix, idf_values = apply_idf(term_doc_matrix)
        idf_time = time.time() - idf_start
        print(f"TIME: {idf_time:.2f}s\n")

    print("Normalizing term-document matrix...")
    norm_start = time.time()
    normalized_matrix = normalize(term_doc_matrix, axis=1)
    norm_time = time.time() - norm_start
    print(f"TIME: {norm_time:.2f}s\n")

    svd_components = None
    if use_svd:
        svd_components = load_or_compute_svd(normalized_matrix, k_svd)

    setup_time = time.time() - dict_start

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


def main():
    """Main search engine function."""
    (
        use_svd,
        normalized_matrix,
        svd_components,
        vocabulary,
        doc_ids,
        idf_values,
        k_results,
    ) = setup()

    while True:
        query = input("\nSearch input (type 'exit' to leave): ").strip()

        if query.lower() == "exit":
            break

        if not query:
            print("Please enter a search query.")
            continue

        res_start = time.time()
        if use_svd:
            results = search_documents_svd(
                query,
                svd_components,
                vocabulary,
                doc_ids,
                idf_values,
                k_results,
            )
        else:
            results = search_documents(
                query,
                normalized_matrix,
                vocabulary,
                doc_ids,
                idf_values,
                k_results,
            )
        res_time = time.time() - res_start

        if results and results[0][1] > 0.001:
            print(f"\nSearch results ({res_time:.2f}s):")
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()

            for doc_id, similarity in results:
                if similarity > 0.001:
                    cursor.execute(
                        "SELECT title, url FROM pages WHERE id = ?", (doc_id,)
                    )
                    result = cursor.fetchone()
                    print(
                        f"(similarity: {similarity:.4f}) {result[0]}, URL: {result[1]}"
                    )

            conn.close()
        else:
            print("\nNo results match your query.")


if __name__ == "__main__":
    main()
