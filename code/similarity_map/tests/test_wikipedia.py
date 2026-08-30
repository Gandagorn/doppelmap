import json

from similarity_map.pipeline.wikipedia import _photo_url_from_api_response, fetch_photos


def test_photo_url_from_api_response_extracts_thumbnail():
    data = {
        "query": {
            "pages": {
                "12345": {
                    "title": "Tom Hanks",
                    "thumbnail": {
                        "source": "https://upload.wikimedia.org/x.jpg",
                        "width": 192,
                        "height": 256,
                    },
                }
            }
        }
    }
    assert _photo_url_from_api_response(data) == "https://upload.wikimedia.org/x.jpg"


def test_photo_url_from_api_response_returns_none_when_no_thumbnail():
    data = {"query": {"pages": {"-1": {"title": "Nonexistent Person", "missing": ""}}}}
    assert _photo_url_from_api_response(data) is None


def test_fetch_photos_uses_injected_fetcher_and_writes_cache(tmp_path):
    cache_path = tmp_path / "wikipedia_photo_cache.json"
    calls = []

    def fake_fetch(name):
        calls.append(name)
        if name == "No Photo Person":
            return None
        return f"https://example.com/{name.replace(' ', '_')}.jpg"

    result = fetch_photos(
        ["Tom Hanks", "No Photo Person"], cache_path=cache_path, fetch_one=fake_fetch
    )

    assert result == {
        "Tom Hanks": "https://example.com/Tom_Hanks.jpg",
        "No Photo Person": None,
    }
    assert calls == ["Tom Hanks", "No Photo Person"]
    assert cache_path.exists()
    on_disk = json.loads(cache_path.read_text(encoding="utf-8"))
    assert on_disk == result


def test_fetch_photos_skips_network_for_cached_names(tmp_path):
    cache_path = tmp_path / "wikipedia_photo_cache.json"
    cache_path.write_text(
        json.dumps({"Tom Hanks": "https://example.com/cached.jpg"}), encoding="utf-8"
    )

    def fail_fetch(name):
        raise AssertionError(f"should not fetch cached name {name}")

    result = fetch_photos(["Tom Hanks"], cache_path=cache_path, fetch_one=fail_fetch)
    assert result == {"Tom Hanks": "https://example.com/cached.jpg"}


def test_fetch_photos_treats_fetch_error_as_no_photo(tmp_path):
    cache_path = tmp_path / "wikipedia_photo_cache.json"

    def erroring_fetch(name):
        raise OSError("network unreachable")

    result = fetch_photos(["Tom Hanks"], cache_path=cache_path, fetch_one=erroring_fetch)
    assert result == {"Tom Hanks": None}
