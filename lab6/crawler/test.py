"""
Crawler test module.
"""
import sqlite3
import os

directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(directory, 'simplified_wiki_index.db')


def search_index(query, limit=10):
    """
    Simple search implementation for testing purposes.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute('''
    SELECT id, url, title, substr(content, 1, 200) 
    FROM pages 
    WHERE title LIKE ? OR content LIKE ?
    LIMIT ?
    ''', (f'%{query}%', f'%{query}%', limit))

    results = cursor.fetchall()
    conn.close()

    formatted_results = []
    for id_, url, title, snippet in results:
        formatted_results.append({
            'id': id_,
            'url': url,
            'title': title,
            'snippet': snippet + '...' if len(snippet) >= 200 else snippet
        })

    return formatted_results


if __name__ == "__main__":
    while True:
        print("Enter a search query (or 'exit' to quit):")
        query_ = input()
        if query_ == "exit":
            break
        results_ = search_index(query_)
        for result in results_:
            print(f"Title: {result['title']}, URL: {result['url']}, Snippet: {result['snippet']}")
