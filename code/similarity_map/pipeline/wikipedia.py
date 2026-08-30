"""Fetches real profile-photo URLs for curated celebrity names from
Wikipedia's public MediaWiki API, with an on-disk cache so re-running the
generator doesn't re-hit the network for names already resolved.
"""
import json
import time
import urllib.parse
import urllib.request
from collections.abc import Callable
from pathlib import Path

API_URL = "https://en.wikipedia.org/w/api.php"
THUMB_WIDTH = 192  # 2x the 96px display size, for retina
REQUEST_DELAY_SECONDS = 0.3  # be polite to the shared, unauthenticated API
# Wikimedia's User-Agent policy asks for a project identifier and contact URL
# to avoid being throttled/blocked as an anonymous client.
USER_AGENT = "doppelmap-dataset-generator/1.0 (https://github.com/Gandagorn/doppelmap)"


def _photo_url_from_api_response(data: dict) -> str | None:
    """Pure parser for a MediaWiki API search-generator response: returns
    the top result's thumbnail URL, or None if there's no match or the
    matched page has no image.
    """
    pages = data.get("query", {}).get("pages", {})
    for page in pages.values():
        thumbnail = page.get("thumbnail")
        if thumbnail and "source" in thumbnail:
            return thumbnail["source"]
    return None


def _fetch_photo_url(name: str) -> str | None:
    # Full-text search (rather than an exact `titles=` match) so names
    # missing diacritics -- our curated list is plain ASCII, e.g. "Beyonce"
    # for the page "Beyonce" -- still resolve to the right page.
    params = {
        "action": "query",
        "generator": "search",
        "gsrsearch": name,
        "gsrlimit": "1",
        "prop": "pageimages",
        "format": "json",
        "pithumbsize": str(THUMB_WIDTH),
    }
    url = f"{API_URL}?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    time.sleep(REQUEST_DELAY_SECONDS)  # be polite to the shared, unauthenticated API
    with urllib.request.urlopen(request, timeout=10) as response:
        data = json.loads(response.read().decode("utf-8"))
    return _photo_url_from_api_response(data)


def fetch_photos(
    names: list[str],
    *,
    cache_path: Path,
    fetch_one: Callable[[str], str | None] = _fetch_photo_url,
) -> dict[str, str | None]:
    """Returns name -> photo URL (or None if no photo was found for that
    name, including on a network/lookup error -- a best-effort visual
    upgrade shouldn't fail the whole dataset build).
    """
    cache: dict[str, str | None] = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))

    updated = False
    for name in names:
        if name in cache:
            continue
        try:
            cache[name] = fetch_one(name)
        except Exception:
            cache[name] = None
        updated = True

    if updated:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache, indent=2, sort_keys=True), encoding="utf-8")

    return {name: cache.get(name) for name in names}
