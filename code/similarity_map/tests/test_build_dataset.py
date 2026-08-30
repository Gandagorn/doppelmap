import json

import numpy as np
import pytest

from similarity_map.pipeline.build_dataset import build_dataset, build_dataset_from_embeddings


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
        assert node["attr"] == "Synthetic placeholder — no real photo"
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


def _write_fake_prototypes_npz(path, names):
    rng = np.random.default_rng(7)
    e = rng.normal(size=(len(names), 512)).astype(np.float32)
    e /= np.linalg.norm(e, axis=1, keepdims=True)
    np.savez(path, names=np.array(names), E=e, n_used=np.full(len(names), 10))


def test_build_dataset_from_embeddings_end_to_end(tmp_path):
    names = [f"Real Person {i}" for i in range(10)]
    npz_path = tmp_path / "prototypes.npz"
    _write_fake_prototypes_npz(npz_path, names)

    graph = build_dataset_from_embeddings(npz_path, k=4, seed=1, out_dir=tmp_path / "out")

    assert graph["meta"]["count"] == 10
    assert graph["meta"]["k"] == 4
    assert len(graph["nodes"]) == 10
    assert set(n["name"] for n in graph["nodes"]) == set(names)

    for node in graph["nodes"]:
        assert set(node.keys()) == {"id", "name", "x", "y", "deg", "thumb", "attr"}
        assert node["attr"] == "Placeholder avatar — real photo not yet linked"
        thumb_path = tmp_path / "out" / node["thumb"]
        assert thumb_path.exists()

    graph_json_path = tmp_path / "out" / "graph.json"
    on_disk = json.loads(graph_json_path.read_text(encoding="utf-8"))
    assert on_disk == graph


def test_build_dataset_from_embeddings_uses_all_names_no_count_cap(tmp_path):
    names = [f"Person {i}" for i in range(37)]
    npz_path = tmp_path / "prototypes.npz"
    _write_fake_prototypes_npz(npz_path, names)

    graph = build_dataset_from_embeddings(npz_path, k=4, seed=1, out_dir=tmp_path / "out")

    assert graph["meta"]["count"] == 37
