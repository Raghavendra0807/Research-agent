
from typing import Optional
from urllib.parse import quote

import requests

SEARCH_API_URL = "https://en.wikipedia.org/w/rest.php/v1/search/page"
SUMMARY_API_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/"
DEFAULT_TIMEOUT = 10
HEADERS = {
    "User-Agent": "ResearchAgent/0.2 (educational research project; contact: github.com/Raghavendra0807/Research-agent)"
}

_shared_session: Optional[requests.Session] = None


def get_wikimedia_session() -> requests.Session:
    """Retrieve or initialize a reusable requests.Session with default headers."""
    global _shared_session
    if _shared_session is None:
        _shared_session = requests.Session()
        _shared_session.headers.update(HEADERS)
    return _shared_session


def search_wikipedia(
    topic: str,
    limit: int = 3,
    session: Optional[requests.Session] = None,
    timeout: int = DEFAULT_TIMEOUT,
) -> list[dict]:
    """
    Find articles and retrieve their lead summaries with robust error handling.
    
    Resilience guarantees:
    - Reuses a shared HTTP session.
    - Handles network timeouts and connection drops cleanly.
    - If a single article summary lookup 404s or fails, it does NOT abort the search.
    - Sources without valid extracted text are not treated as valid evidence.
    """
    cleaned_topic = topic.strip() if topic else ""
    if not cleaned_topic or limit <= 0:
        return []

    active_session = session or get_wikimedia_session()

    # 1. Search Wikipedia for matching article titles
    try:
        response = active_session.get(
            SEARCH_API_URL,
            params={"q": cleaned_topic, "limit": limit},
            timeout=timeout,
        )
        response.raise_for_status()
        search_data = response.json()
    except (requests.exceptions.RequestException, ValueError) as error:
        # Search endpoint failure (timeout, network error, HTTP 5xx, or non-JSON)
        raise requests.exceptions.RequestException(
            f"Wikipedia search endpoint failed for '{cleaned_topic}': {error}"
        ) from error

    pages = search_data.get("pages", []) if isinstance(search_data, dict) else []
    if not isinstance(pages, list):
        return []

    sources = []

    # 2. Fetch full lead summary for each returned article
    for page in pages:
        if not isinstance(page, dict):
            continue

        title = page.get("title", "")
        if not isinstance(title, str) or not title.strip():
            continue

        title = title.strip()
        encoded_title = quote(title.replace(" ", "_"), safe="()")
        summary_url = f"{SUMMARY_API_URL}{encoded_title}"

        try:
            summary_response = active_session.get(summary_url, timeout=timeout)

            # 404 means summary does not exist for this title; skip to next article
            if summary_response.status_code == 404:
                continue

            summary_response.raise_for_status()
            summary = summary_response.json()
        except (requests.exceptions.RequestException, ValueError):
            # A failure in fetching a single summary shouldn't crash the entire run
            continue

        if not isinstance(summary, dict):
            continue

        # Extract text: only accept non-empty text as valid evidence
        extract = summary.get("extract", "")
        if not isinstance(extract, str) or not extract.strip():
            continue

        # Canonical desktop URL extraction with reliable fallback
        desktop_url = ""
        content_urls = summary.get("content_urls")
        if isinstance(content_urls, dict):
            desktop_info = content_urls.get("desktop")
            if isinstance(desktop_info, dict):
                desktop_url = desktop_info.get("page", "")

        if not desktop_url:
            desktop_url = f"https://en.wikipedia.org/wiki/{encoded_title}"

        sources.append({
            "title": title,
            "url": desktop_url,
            "text": extract.strip(),
        })

    return sources


if __name__ == "__main__":
    test_queries = [
        "food redistribution",
        "food bank",
        "CRISPR gene editing",
    ]

    seen = set()
    all_sources = []

    for query in test_queries:
        print(f"\nSearching: {query}")
        try:
            results = search_wikipedia(query, limit=2)

            for src in results:
                t = src.get("title", "").strip()
                k = t.casefold()

                if t and k not in seen:
                    seen.add(k)
                    all_sources.append(src)
                    print(f"  - {t}: {src.get('url', '')}")

        except Exception as exc:
            print(f"  Search failed: {exc}")

    print(f"\nCollected {len(all_sources)} unique sources.")


