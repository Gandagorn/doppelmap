import json

import pytest

from similarity_map.pipeline.build_dataset import build_dataset


def test_build_dataset_end_to_end(tmp_path):
    graph = build_dataset(count=12, k=4, seed=1, out_dir=tmp_path)

    assert graph["meta"]["count"] == 12
    assert graph["meta"]["k"] == 4
    assert len(graph["nodes"]) == 12

    graph_json_path = tmp_path / "graph.json"
    assert graph_json_path.exists()
    on_disk = json.loads(graph_json_path.read_text(encoding="utf-8"))
    assert on_disk == graph

    for node in graph["nodes"]:
        assert set(node.keys()) == {"id", "name", "x", "y", "deg", "thumb", "attr"}
        thumb_path = tmp_path / node["thumb"]
        assert thumb_path.exists()

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
