import json

import pytest

from similarity_map.pipeline.build_dataset import build_dataset


def _make_fake_fetch_photo():
    # Deterministic stand-in for the real Wikipedia lookup: alternates
    # found/not-found per call, independent of which names are actually
    # curated, so the test exercises both the found and fallback code paths.
    calls = {"n": 0}

    def fake_fetch_photo(name):
        calls["n"] += 1
        if calls["n"] % 2 == 0:
            return None
        return f"https://example.com/{name.replace(' ', '_')}.jpg"

    return fake_fetch_photo


def test_build_dataset_end_to_end(tmp_path):
    graph = build_dataset(
        count=12,
        k=4,
        seed=1,
        out_dir=tmp_path,
        fetch_photo=_make_fake_fetch_photo(),
        photo_cache_path=tmp_path / "photo_cache.json",
    )

    assert graph["meta"]["count"] == 12
    assert graph["meta"]["k"] == 4
    assert len(graph["nodes"]) == 12

    graph_json_path = tmp_path / "graph.json"
    assert graph_json_path.exists()
    on_disk = json.loads(graph_json_path.read_text(encoding="utf-8"))
    assert on_disk == graph

    found_photo = False
    found_no_photo = False
    for node in graph["nodes"]:
        assert set(node.keys()) == {"id", "name", "x", "y", "deg", "thumb", "photo", "attr"}
        thumb_path = tmp_path / node["thumb"]
        assert thumb_path.exists()
        if node["photo"] is not None:
            assert node["photo"] == f"https://example.com/{node['name'].replace(' ', '_')}.jpg"
            assert node["attr"] == "Photo: Wikipedia"
            found_photo = True
        else:
            assert node["attr"] == "Synthetic placeholder — no real photo"
            found_no_photo = True
    assert found_photo, "expected at least one node to get a fake photo URL"
    assert found_no_photo, "expected at least one node to fall back to the placeholder"

    for a, b, w in graph["edges"]:
        assert 0 <= a < 12
        assert 0 <= b < 12
        assert 0.0 <= w <= 1.0

    assert set(graph["similar"].keys()) == {str(i) for i in range(12)}


def test_build_dataset_rejects_count_over_available_names():
    with pytest.raises(ValueError):
        build_dataset(count=100_000, k=4, seed=1, out_dir=None)


def test_build_dataset_rejects_non_positive_count():
    with pytest.raises(ValueError):
        build_dataset(count=0, k=4, seed=1, out_dir=None)
