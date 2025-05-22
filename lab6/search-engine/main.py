"""Main module for the search engine."""

import os
import sqlite3
import time

from src.setup import setup
from src.search import search_documents, search_documents_svd

DB_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "data",
    "simplified_wiki_index.db",
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
