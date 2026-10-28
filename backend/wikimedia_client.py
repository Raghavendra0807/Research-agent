import re
import time
from typing import Callable, Optional
from urllib.parse import quote

import requests

SEARCH_API_URL = "https://en.wikipedia.org/w/rest.php/v1/search/page"
SUMMARY_API_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/"
DEFAULT_TIMEOUT = 10
DEFAULT_MAX_RETRIES = 2
DEFAULT_INITIAL_BACKOFF = 0.1
DEFAULT_BACKOFF_FACTOR = 2.0
DEFAULT_MAX_BACKOFF = 8.0
DEFAULT_PACING_DELAY = 0.05

HEADERS = {
    "User-Agent": (
        "ResearchAgent/0.2 (https://github.com/Raghavendra0807/Research-agent; "
        "educational research briefing agent) requests/2.32"
    )
}

_shared_session: Optional[requests.Session] = None
_summary_cache: dict[str, dict] = {}


class WikimediaRateLimitError(requests.exceptions.RequestException):
    """Raised when Wikimedia search or summary endpoints return HTTP 429 and retries are exhausted."""

    def __init__(
        self,
        message: str,
        retry_after: Optional[float] = None,
        response: Optional[requests.Response] = None,
    ):
        super().__init__(message, response=response)
        self.retry_after = retry_after


def is_rate_limit_error(error: Exception) -> bool:
    """Determine whether an exception represents an HTTP 429 rate limit condition."""
    if isinstance(error, WikimediaRateLimitError):
        return True
    response = getattr(error, "response", None)
    if response is not None and getattr(response, "status_code", None) == 429:
        return True
    err_str = str(error).lower()
    return "429" in err_str and ("rate" in err_str or "too many requests" in err_str)


def clear_summary_cache() -> None:
    """Clear the in-memory Wikipedia article summary cache."""
    global _summary_cache
    _summary_cache.clear()


def parse_retry_after(header_val: Optional[str], default_backoff: float) -> float:
    """Safely parse Retry-After header (seconds integer or float), falling back to default."""
    if not header_val:
        return default_backoff
    try:
        val = float(header_val.strip())
        return max(0.01, val)
    except (ValueError, TypeError):
        return default_backoff


STOP_WORDS = {
    "a", "an", "the", "and", "or", "of", "in", "on", "at", "to", "for", "with",
    "by", "from", "up", "about", "into", "over", "after", "is", "are", "was",
    "were", "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "how", "what", "which", "who", "whom", "this", "that", "these", "those",
    "their", "its", "it", "they", "them", "within", "between", "under", "both",
    "some", "such", "than", "too", "very", "can", "could", "would", "should",
    "ways", "terms", "manner", "nature", "explain", "describe",
}

QUESTION_PREFIX_REGEX = re.compile(
    r"^(what\s+(is|are|role|key|documented|mechanisms)|how\s+(does|do|is|are|can)|in\s+what\s+ways\s+(does|do)|why\s+(does|do|is)|explain\s+(how|what|the)|describe\s+(how|what|the))\b",
    re.IGNORECASE,
)


def derive_search_query(question: str, topic: str = "") -> str:
    """
    Derive a concise topic-focused Wikipedia search query from a research question.
    Strips question prefixes, conversational filler, and parentheticals while
    preserving core domain concepts and ensuring topic keywords are present.
    """
    cleaned = question.strip() if question else ""
    if not cleaned:
        return topic.strip()

    # If already very short, preserve as-is for unit test compatibility
    if len(cleaned.split()) <= 3:
        return cleaned

    # 1. Strip parentheticals like (e.g., ...)
    no_parens = re.sub(r"\([^)]*\)", " ", cleaned)
    # 2. Strip question mark and non-alphanumeric punctuation (except hyphens)
    no_punct = re.sub(r"[^\w\s\-]", " ", no_parens)
    normalized = re.sub(r"\s+", " ", no_punct).strip()

    # 3. Strip leading question prefixes
    without_prefix = QUESTION_PREFIX_REGEX.sub("", normalized).strip()

    # 4. Tokenize
    tokens = [w for w in without_prefix.split() if w]

    # Filter filler/stop words while preserving meaningful technical words
    meaningful = [
        w for w in tokens
        if w.lower() not in STOP_WORDS and len(w) > 1
    ]

    # 5. Ensure topic terms are included
    topic_clean = topic.strip()
    topic_tokens = [w for w in topic_clean.split() if w]

    query_str_lower = " ".join(meaningful).lower()

    if topic_clean:
        missing_topic = [w for w in topic_tokens if w.lower() not in query_str_lower]
        if missing_topic:
            meaningful = missing_topic + meaningful

    # Cap at 5 words
    final_tokens = meaningful[:5]
    if not final_tokens:
        return topic_clean or cleaned

    return " ".join(final_tokens)


def get_wikimedia_session() -> requests.Session:
    """Retrieve or initialize a reusable requests.Session with default headers."""
    global _shared_session
    if _shared_session is None:
        _shared_session = requests.Session()
        _shared_session.headers.update(HEADERS)
    return _shared_session


def make_wikimedia_request(
    url: str,
    params: Optional[dict] = None,
    session: Optional[requests.Session] = None,
    timeout: int = DEFAULT_TIMEOUT,
    max_retries: int = DEFAULT_MAX_RETRIES,
    initial_backoff: float = DEFAULT_INITIAL_BACKOFF,
    backoff_factor: float = DEFAULT_BACKOFF_FACTOR,
    max_backoff: float = DEFAULT_MAX_BACKOFF,
    pacing_delay: float = DEFAULT_PACING_DELAY,
    sleep_fn: Callable[[float], None] = time.sleep,
    retry_on_timeout: bool = True,
) -> requests.Response:
    """
    Execute an HTTP GET request to Wikimedia endpoints with:
    - Polite outbound request pacing.
    - Bounded retries with exponential backoff on HTTP 429 and transient 5xx server errors.
    - Support for 'Retry-After' header when supplied by Wikimedia.
    - Rate limit failure diagnosis via WikimediaRateLimitError.
    """
    active_session = session or get_wikimedia_session()

    if pacing_delay > 0:
        sleep_fn(pacing_delay)

    for attempt in range(max_retries + 1):
        try:
            response = active_session.get(url, params=params, timeout=timeout)
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as exc:
            if attempt < max_retries and retry_on_timeout:
                backoff = min(initial_backoff * (backoff_factor ** attempt), max_backoff)
                sleep_fn(backoff)
                continue
            raise requests.exceptions.RequestException(
                f"Wikimedia network connection failed for '{url}': {exc}"
            ) from exc

        # Handle HTTP 429 Too Many Requests
        if response.status_code == 429:
            retry_after_hdr = response.headers.get("Retry-After")
            computed_backoff = min(initial_backoff * (backoff_factor ** attempt), max_backoff)
            delay = parse_retry_after(retry_after_hdr, computed_backoff)
            delay = min(delay, max_backoff)

            if attempt < max_retries:
                sleep_fn(delay)
                continue
            else:
                raise WikimediaRateLimitError(
                    f"Wikipedia search endpoint rate limited (HTTP 429) for '{url}' after {max_retries} retries.",
                    retry_after=delay,
                    response=response,
                )

        # Handle transient 5xx server errors (500, 502, 503, 504)
        if response.status_code in {500, 502, 503, 504}:
            if attempt < max_retries:
                backoff = min(initial_backoff * (backoff_factor ** attempt), max_backoff)
                sleep_fn(backoff)
                continue
            else:
                response.raise_for_status()

        return response

    raise requests.exceptions.RequestException(f"Wikimedia request retries exhausted for '{url}'.")


def search_wikipedia(
    topic: str,
    limit: int = 3,
    session: Optional[requests.Session] = None,
    timeout: int = DEFAULT_TIMEOUT,
    max_retries: int = DEFAULT_MAX_RETRIES,
    sleep_fn: Callable[[float], None] = time.sleep,
    pacing_delay: float = DEFAULT_PACING_DELAY,
) -> list[dict]:
    """
    Find articles and retrieve their lead summaries with robust error handling.

    Resilience guarantees:
    - Reuses a shared HTTP session.
    - Applies polite outbound pacing and bounded retries with backoff.
    - Distinguishes HTTP 429 rate-limiting from standard failures.
    - Respects 'Retry-After' when supplied by Wikipedia.
    - Caches article summaries in memory to prevent redundant queries.
    - If a single article summary lookup 404s or fails, it does NOT abort the search.
    - Sources without valid extracted text are not treated as valid evidence.
    """
    cleaned_topic = topic.strip() if topic else ""
    if not cleaned_topic or limit <= 0:
        return []

    active_session = session or get_wikimedia_session()

    # 1. Search Wikipedia for matching article titles
    try:
        response = make_wikimedia_request(
            SEARCH_API_URL,
            params={"q": cleaned_topic, "limit": limit},
            session=active_session,
            timeout=timeout,
            max_retries=max_retries,
            pacing_delay=pacing_delay,
            sleep_fn=sleep_fn,
        )
        response.raise_for_status()
        search_data = response.json()
    except WikimediaRateLimitError:
        raise
    except (requests.exceptions.RequestException, ValueError) as error:
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
        cache_key = encoded_title.casefold()

        # Check summary cache to eliminate redundant HTTP requests across queries
        if cache_key in _summary_cache:
            sources.append(_summary_cache[cache_key])
            continue

        summary_url = f"{SUMMARY_API_URL}{encoded_title}"

        try:
            summary_response = make_wikimedia_request(
                summary_url,
                session=active_session,
                timeout=timeout,
                max_retries=max_retries,
                pacing_delay=pacing_delay,
                sleep_fn=sleep_fn,
                retry_on_timeout=False,
            )

            # 404 means summary does not exist for this title; skip to next article
            if summary_response.status_code == 404:
                continue

            summary_response.raise_for_status()
            summary = summary_response.json()
        except WikimediaRateLimitError:
            continue
        except (requests.exceptions.RequestException, ValueError):
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

        source_obj = {
            "title": title,
            "url": desktop_url,
            "text": extract.strip(),
        }
        _summary_cache[cache_key] = source_obj
        sources.append(source_obj)

    return sources
