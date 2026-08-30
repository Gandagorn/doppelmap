import json

import numpy as np
import pytest

from similarity_map.pipeline.build_dataset import build_dataset, build_dataset_from_embeddings


def test_build_dataset_end_to_end(tmp_path):
    graph = build_dataset(count=12, k=4, seed=1, out_dir=tmp_path)

    assert graph["meta"]["count"] == 12
    assert graph["meta"]["k"] == 4
    assert len(graph["nodes"]) == 12

    graph_json_path = tmp_path / "graph-all.json"
    assert graph_json_path.exists()
    on_disk = json.loads(graph_json_path.read_text(encoding="utf-8"))
    assert on_disk == graph

    for node in graph["nodes"]:
        assert set(node.keys()) == {"id", "name", "x", "y", "deg", "thumb", "attr", "popularity"}
        assert node["attr"] == "Synthetic placeholder — no real photo"
        assert node["popularity"] == 1
        thumb_path = tmp_path / node["thumb"]
        assert thumb_path.exists()

    for a, b, w in graph["edges"]:
        assert 0 <= a < 12
        assert 0 <= b < 12
        assert 0.0 <= w <= 1.0

    assert set(graph["similar"].keys()) == {str(i) for i in range(12)}


def test_build_dataset_writes_identical_copies_for_every_level(tmp_path):
    # Synthetic mode has no meaningful popularity variation to split levels
    # on, but the frontend always expects one file per level -- so all four
    # must exist, with the same content, to avoid a 404 when switching.
    graph = build_dataset(count=8, k=4, seed=1, out_dir=tmp_path)

    for label in ("all", "top50", "top20", "top5"):
        on_disk = json.loads((tmp_path / f"graph-{label}.json").read_text(encoding="utf-8"))
        assert on_disk == graph


def test_build_dataset_rejects_count_over_available_names():
    with pytest.raises(ValueError):
        build_dataset(count=100_000, k=4, seed=1, out_dir=None)


def test_build_dataset_rejects_non_positive_count():
    with pytest.raises(ValueError):
        build_dataset(count=0, k=4, seed=1, out_dir=None)


def _write_fake_prototypes_npz(path, names, n_used=None):
    rng = np.random.default_rng(7)
    e = rng.normal(size=(len(names), 512)).astype(np.float32)
    e /= np.linalg.norm(e, axis=1, keepdims=True)
    if n_used is None:
        n_used = np.full(len(names), 10)
    np.savez(path, names=np.array(names), E=e, n_used=n_used)


def test_build_dataset_from_embeddings_returns_one_graph_per_level(tmp_path):
    # Varied popularity (1..100) so each percentile level genuinely differs
    # in size, unlike a uniform fixture where every level would include
    # everyone.
    names = [f"Person {i}" for i in range(100)]
    n_used = np.arange(1, 101)  # person i has popularity i+1
    npz_path = tmp_path / "prototypes.npz"
    _write_fake_prototypes_npz(npz_path, names, n_used=n_used)

    graphs = build_dataset_from_embeddings(npz_path, k=4, seed=1, out_dir=tmp_path / "out")

    assert set(graphs.keys()) == {"all", "top50", "top20", "top5"}
    # MIN_IMAGES=5 drops persons 0-3 (popularity 1-4), leaving 96 for "all"
    assert graphs["all"]["meta"]["count"] == 96
    # each stricter level must be a strictly smaller (or equal) subset
    assert graphs["top50"]["meta"]["count"] < graphs["all"]["meta"]["count"]
    assert graphs["top20"]["meta"]["count"] < graphs["top50"]["meta"]["count"]
    assert graphs["top5"]["meta"]["count"] < graphs["top20"]["meta"]["count"]

    for label in ("all", "top50", "top20", "top5"):
        graph_json_path = tmp_path / "out" / f"graph-{label}.json"
        assert graph_json_path.exists()
        on_disk = json.loads(graph_json_path.read_text(encoding="utf-8"))
        assert on_disk == graphs[label]
        for node in graphs[label]["nodes"]:
            assert set(node.keys()) == {"id", "name", "x", "y", "deg", "thumb", "attr", "popularity"}
            thumb_path = tmp_path / "out" / node["thumb"]
            assert thumb_path.exists()


def test_build_dataset_from_embeddings_shares_thumbnails_across_levels(tmp_path):
    # The same person appears in multiple levels (e.g. everyone in "top5" is
    # also in "all") -- thumbnails must be generated once and shared by
    # filename, not duplicated once per level.
    names = [f"Person {i}" for i in range(50)]
    n_used = np.arange(1, 51)
    npz_path = tmp_path / "prototypes.npz"
    _write_fake_prototypes_npz(npz_path, names, n_used=n_used)

    graphs = build_dataset_from_embeddings(npz_path, k=4, seed=1, out_dir=tmp_path / "out")

    thumb_paths_by_name = {}
    for graph in graphs.values():
        for node in graph["nodes"]:
            if node["name"] in thumb_paths_by_name:
                assert thumb_paths_by_name[node["name"]] == node["thumb"]
            else:
                thumb_paths_by_name[node["name"]] = node["thumb"]

    thumbs_dir = tmp_path / "out" / "thumbs"
    assert thumbs_dir.exists()
    # one file per unique surviving name, not per (name, level) pair
    assert len(list(thumbs_dir.iterdir())) == len(thumb_paths_by_name)


def test_build_dataset_from_embeddings_uses_all_names_no_count_cap(tmp_path):
    names = [f"Person {i}" for i in range(37)]
    npz_path = tmp_path / "prototypes.npz"
    _write_fake_prototypes_npz(npz_path, names)

    graphs = build_dataset_from_embeddings(npz_path, k=4, seed=1, out_dir=tmp_path / "out")

    assert graphs["all"]["meta"]["count"] == 37


def test_build_dataset_from_embeddings_drops_thin_prototypes(tmp_path):
    names = [f"Person {i}" for i in range(10)] + ["Thin Person"]
    n_used = np.array([10] * 10 + [2])  # "Thin Person" is below MIN_IMAGES
    npz_path = tmp_path / "prototypes.npz"
    _write_fake_prototypes_npz(npz_path, names, n_used=n_used)

    graphs = build_dataset_from_embeddings(npz_path, k=4, seed=1, out_dir=tmp_path / "out")

    assert graphs["all"]["meta"]["count"] == 10
    assert "Thin Person" not in {n["name"] for n in graphs["all"]["nodes"]}
