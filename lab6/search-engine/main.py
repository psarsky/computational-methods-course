"""Main module for the search engine."""

import os
import sqlite3

from sklearn.preprocessing import normalize
from src.preprocessing import (apply_idf, build_vocabulary,
                               create_document_vectors)
from src.search import search_documents


def main():
    """Main search engine function."""

    DB_PATH = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "data",
        "simplified_wiki_index.db",
    )

    custom_db = input("Database path (default: [project directory]/search_engine/data/simplified_wiki_index.db): ")
    db_path = custom_db if custom_db else DB_PATH

    use_idf_input = input("Use IDF (y/n, default: n): ").lower()
    use_idf = use_idf_input.startswith("y")

    k_results_input = input("Amount of results to return (default: 10): ")
    k_results = int(k_results_input) if k_results_input.isdigit() else 10

    print("Building term dictionary...")
    vocabulary = build_vocabulary(db_path)

    print("Creating term-document matrix...")
    term_doc_matrix, doc_ids = create_document_vectors(db_path, vocabulary)

    idf_values = None
    if use_idf:
        print("Applying IDF transformation...")
        term_doc_matrix, idf_values = apply_idf(term_doc_matrix)

    print("Normalizing term-document matrix...")
    normalized_matrix = normalize(term_doc_matrix, axis=1)

    while True:
        query = input("\nSearch input (type 'exit' to leave): ")
        if query.lower() == "exit":
            break

        results = search_documents(
            query,
            normalized_matrix,
            vocabulary,
            doc_ids,
            idf_values,
            k=k_results,
        )
        print("Search results:")

        if results:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            for doc_id, similarity in results:
                cursor.execute("SELECT title, url FROM pages WHERE id = ?", (doc_id,))
                result = cursor.fetchone()
                print(f"(similarity: {similarity:.4f}) {result[0]}, URL: {result[1]}")

            conn.close()


if __name__ == "__main__":
    main()
