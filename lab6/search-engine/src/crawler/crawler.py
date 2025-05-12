"""Crawler module to fetch and parse HTML content from a given URL."""

import time
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from database import get_stats, init_database, store_page

session = requests.Session()
BASE_URL = "https://simple.wikipedia.org"
START_URL = "https://simple.wikipedia.org/w/index.php?title=Special:AllPages&from=Court+%28law%29"
REQUEST_DELAY = 1


def is_valid_wiki_page(url):
    """Check if URL is a valid Wikipedia article page.

    Args:
        url: URL to check

    Returns:
        bool: True if valid article page, False otherwise
    """
    if not url.startswith(BASE_URL):
        return False

    excluded_patterns = [
        "/wiki/File:",
        "/wiki/Special:",
        "/wiki/Template:",
        "/wiki/Category:",
        "/wiki/Help:",
        "/wiki/User:",
        "/wiki/Wikipedia:",
        "/wiki/Talk:",
        "/wiki/Template_talk:",
        "/w/index.php",
        "/wiki/Portal:",
    ]

    for pattern in excluded_patterns:
        if pattern in url:
            return False

    return "/wiki/" in url


def normalize_url(url):
    """Normalize URL to ensure consistent format.

    Args:
        url: URL to normalize

    Returns:
        str: Normalized URL
    """
    url = url.split("#")[0]
    url = url.split("?")[0]
    return url


def extract_content(soup):
    """Extract relevant content from a Wikipedia page.

    Args:
        soup: BeautifulSoup object

    Returns:
        tuple: (title, content)
    """
    title_element = soup.find("h1", id="firstHeading")
    title = title_element.text.strip() if title_element else "Unknown"

    content_div = soup.find("div", id="mw-content-text")

    if content_div:
        for unwanted in content_div.select(
            ".navbox, .ambox, .mbox-small, .reference, .mw-editsection"
        ):
            if unwanted:
                unwanted.decompose()

        paragraphs = [
            p.text.strip() for p in content_div.find_all("p") if p.text.strip()
        ]
        content = "\n\n".join(paragraphs)
    else:
        content = ""

    return title, content


def get_urls_from_special_page(url):
    """Get URLs from a Special:AllPages page.

    Args:
        url: URL of the Special:AllPages page

    Returns:
        tuple: (article_urls, next_page_url or None)
    """
    response = session.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    urls = []
    content_div = soup.find("div", id="mw-content-text")
    if content_div:
        for li in content_div.select("ul.mw-allpages-chunk li"):
            a_tag = li.find("a")
            if a_tag and "href" in a_tag.attrs:
                article_url = urljoin(BASE_URL, a_tag["href"])
                if is_valid_wiki_page(article_url):
                    urls.append(article_url)

    next_page = None
    nav_div = soup.find("div", {"class": "mw-allpages-nav"})
    if nav_div:
        next_links = [a for a in nav_div.find_all("a") if "Next page" in a.text]
        if next_links:
            next_page = urljoin(BASE_URL, next_links[0]["href"])

    return urls, next_page


def crawl():
    """Main crawler function."""

    visited_urls = set()

    current_special_page = START_URL
    total_pages = 0
    page_no = 0
    errors = 0

    while current_special_page:
        print(f"Processing index page: {current_special_page}")
        article_urls, next_special_page = get_urls_from_special_page(
            current_special_page
        )

        for article_url in article_urls:
            page_no += 1
            if article_url in visited_urls:
                continue

            try:
                print(f"Page {page_no}: {article_url}")
                response = session.get(article_url)
                soup = BeautifulSoup(response.text, "html.parser")

                title, content = extract_content(soup)
                store_page(article_url, title, content)

                visited_urls.add(article_url)
                total_pages += 1

                if total_pages % 100 == 0:
                    print(f"Total pages crawled: {total_pages}")

                time.sleep(REQUEST_DELAY)

            except Exception as e:
                print(f"Exception: {str(e)}")
                errors += 1

        current_special_page = next_special_page

        time.sleep(REQUEST_DELAY)

    print(f"Indexed pages: {total_pages}")
    print(f"Errors: {errors}")


if __name__ == "__main__":
    init_database()

    try:
        crawl()
    except KeyboardInterrupt:
        print("\nCrawl interrupted by user")

    stats = get_stats()
    print("\nStatistics:")
    print(f"Total pages indexed: {stats['total_pages']}")
    print("Recent pages:")
    for page in stats["recent_pages"]:
        print(f"- {page}")
