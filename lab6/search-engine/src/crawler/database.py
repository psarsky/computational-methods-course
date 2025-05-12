"""
Module for managing the SQLite database used in the web crawler.
"""

import os
import sqlite3

directory = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data"
)
os.makedirs(directory, exist_ok=True)
DB_PATH = os.path.join(directory, "simplified_wiki_index.db")


def init_database():
    """
    Initializes the SQLite database.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(
        """
    CREATE TABLE IF NOT EXISTS pages (
        id INTEGER PRIMARY KEY,
        url TEXT UNIQUE,
        title TEXT,
        content TEXT,
        last_crawled TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """
    )

    cursor.execute("CREATE INDEX IF NOT EXISTS idx_url ON pages (url)")

    conn.commit()
    conn.close()


def store_page(url, title, content):
    """
    Store page information in the database.

    Args:
        url: Page URL
        title: Page title
        content: Page content

    Returns:
        int: Page ID in database
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("SELECT id FROM pages WHERE url = ?", (url,))
    result = cursor.fetchone()

    if result:
        page_id = result[0]
        cursor.execute(
            "UPDATE pages SET title = ?, content = ?, last_crawled = datetime(current_timestamp, 'localtime') \
                WHERE id = ?",
            (title, content, page_id),
        )
    else:
        cursor.execute(
            "INSERT INTO pages (url, title, content, last_crawled) VALUES (?, ?, ?, datetime(current_timestamp, \
                'localtime'))",
            (url, title, content),
        )
        page_id = cursor.lastrowid

    conn.commit()
    conn.close()

    return page_id


def get_stats():
    """
    Get statistics about the crawled data.

    Returns:
        dict: Statistics about the crawl
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM pages")
    page_count = cursor.fetchone()[0]

    cursor.execute("SELECT title FROM pages ORDER BY id DESC LIMIT 5")
    recent_pages = [row[0] for row in cursor.fetchall()]

    conn.close()

    return {"total_pages": page_count, "recent_pages": recent_pages}
