"""Search Engine with Flask API"""

import os
import sqlite3
import threading
import time

import nltk
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from nltk.corpus import stopwords
from sklearn.preprocessing import normalize
from src.preprocessing import apply_idf
from src.search import search_documents, search_documents_svd
from src.setup import (
    load_or_compute_svd,
    load_or_compute_term_doc_matrix,
    load_or_compute_vocabulary,
)


def check_nltk_data():
    """Check if NLTK data is downloaded."""
    try:
        stopwords.words("english")
        print("\nNLTK data already downloaded.")
    except LookupError:
        print("\nDownloading NLTK data...")
        nltk.download("stopwords")
        print("NLTK data downloaded successfully!")


app = Flask(__name__, static_folder="client")
CORS(app)

engine_components = {
    "initialized": False,
    "in_progress": False,
    "vocabulary": None,
    "normalized_matrix": None,
    "svd_components": None,
    "doc_ids": None,
    "idf_values": None,
    "use_svd": True,
    "k_svd": 100,
    "k_results": 10,
}

init_lock = threading.Lock()

DB_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "data",
    "simplified_wiki_index.db",
)


def initialize_engine():
    """Initialize search engine components."""
    with init_lock:
        if engine_components["initialized"] or engine_components["in_progress"]:
            return

        engine_components["in_progress"] = True

        try:
            print("\nStarting search engine initialization...")

            vocabulary = load_or_compute_vocabulary()
            engine_components["vocabulary"] = vocabulary

            term_doc_matrix, doc_ids = load_or_compute_term_doc_matrix(vocabulary)
            engine_components["doc_ids"] = doc_ids

            print("\nApplying IDF...")
            term_doc_matrix, idf_values = apply_idf(term_doc_matrix)
            engine_components["idf_values"] = idf_values

            print("\nNormalizing term-document matrix...")
            normalized_matrix = normalize(term_doc_matrix, axis=1)
            engine_components["normalized_matrix"] = normalized_matrix

            use_svd = engine_components["use_svd"]
            k_svd = engine_components["k_svd"]

            if use_svd:
                svd_components = load_or_compute_svd(True, normalized_matrix, k_svd)
                engine_components["svd_components"] = svd_components

            engine_components["initialized"] = True
            print("\nSearch engine initialization complete!")

        except Exception as e:
            print(f"\nError during initialization: {str(e)}")
        finally:
            engine_components["in_progress"] = False


@app.route("/api/status")
def status():
    """Checks the initialization status of the search engine.

    Returns:
        flask.Response: JSON response with the status of the search engine.
    """
    status_info = {
        "status": "ready" if engine_components["initialized"] else "initializing",
        "in_progress": engine_components["in_progress"],
        "use_svd": engine_components["use_svd"],
        "k_svd": engine_components["k_svd"],
        "k_results": engine_components["k_results"],
    }

    if engine_components["in_progress"]:
        status_info["message"] = "Engine is currently initializing..."
    elif not engine_components["initialized"]:
        status_info["message"] = "Engine needs initialization"
    else:
        status_info["message"] = "Engine is ready"

    return jsonify(status_info)


@app.route("/api/config", methods=["POST"])
def update_config():
    """Update search engine configuration.

    Returns:
        flask.Response: JSON response with the updated configuration or error message.
    """
    if engine_components["in_progress"]:
        return (
            jsonify(
                {
                    "error": "Engine is currently initializing, cannot update configuration"
                }
            ),
            409,
        )

    data = request.get_json()

    if "use_svd" in data:
        engine_components["use_svd"] = bool(data["use_svd"])

    if "k_svd" in data and isinstance(data["k_svd"], int) and data["k_svd"] > 0:
        engine_components["k_svd"] = data["k_svd"]

    if (
        "k_results" in data
        and isinstance(data["k_results"], int)
        and data["k_results"] > 0
    ):
        engine_components["k_results"] = data["k_results"]
    if "use_svd" in data or "k_svd" in data:
        engine_components["initialized"] = False
        threading.Thread(target=initialize_engine).start()

    return jsonify(
        {
            "status": "config_updated",
            "use_svd": engine_components["use_svd"],
            "k_svd": engine_components["k_svd"],
            "k_results": engine_components["k_results"],
        }
    )


@app.route("/api/search")
def search():
    """Search endpoint.

    Returns:
        flask.Response: JSON response with search results or error message.
    """
    query = request.args.get("q", "")

    if not query:
        return jsonify({"error": "Query parameter 'q' is required"}), 400

    if not engine_components["initialized"]:
        if engine_components["in_progress"]:
            return (
                jsonify(
                    {"error": "Search engine is initializing, please try again later"}
                ),
                503,
            )
        threading.Thread(target=initialize_engine).start()
        return (
            jsonify({"error": "Search engine is initializing, please try again later"}),
            503,
        )

    try:
        res_start = time.time()

        k_results = engine_components["k_results"]

        if engine_components["use_svd"]:
            results = search_documents_svd(
                query,
                engine_components["svd_components"],
                engine_components["vocabulary"],
                engine_components["doc_ids"],
                engine_components["idf_values"],
                k_results,
            )
        else:
            results = search_documents(
                query,
                engine_components["normalized_matrix"],
                engine_components["vocabulary"],
                engine_components["doc_ids"],
                engine_components["idf_values"],
                k_results,
            )

        res_time = time.time() - res_start

        formatted_results = []

        if results and results[0][1] > 0.001:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()

            for doc_id, similarity in results:
                if similarity > 0.001:
                    cursor.execute(
                        "SELECT title, url FROM pages WHERE id = ?", (doc_id,)
                    )
                    result = cursor.fetchone()
                    if result:
                        formatted_results.append(
                            {
                                "doc_id": doc_id,
                                "similarity": float(similarity),
                                "title": result[0],
                                "url": result[1],
                            }
                        )

            conn.close()

        return jsonify(
            {
                "query": query,
                "time": res_time,
                "results_count": len(formatted_results),
                "results": formatted_results,
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/document/<int:doc_id>")
def get_document(doc_id):
    """Get document details by ID.

    Args:
        doc_id (int): ID of the document to retrieve.

    Returns:
        flask.Response: JSON response with document details or error message.
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT id, title, url, content FROM pages WHERE id = ?", (doc_id,)
        )
        result = cursor.fetchone()
        conn.close()

        if result:
            return jsonify(
                {
                    "id": result[0],
                    "title": result[1],
                    "url": result[2],
                    "content": result[3],
                }
            )
        return jsonify({"error": f"Document with ID {doc_id} not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/")
def index():
    """Serve the index.html file.

    Returns:
        flask.Response: HTML response with the index page.
    """
    return send_from_directory("client", "index.html")


@app.route("/<path:path>")
def serve_static(path):
    """Serve static files.

    Args:
        path (str): Path to the static file.

    Returns:
        flask.Response: Static file response.
    """
    return send_from_directory("client", path)


if __name__ == "__main__":
    check_nltk_data()
    os.makedirs("data", exist_ok=True)

    print("\nStarting search engine API on http://localhost:5000")
    threading.Thread(target=initialize_engine).start()

    app.run()
