import numpy as np

from similarity_map.pipeline.real_embeddings import (
    filter_prototypes,
    load_popularity,
    load_real_embeddings,
)


def _write_fake_npz(path):
    names = np.array(["Tom Hanks", "Meryl Streep"])
    e = np.random.default_rng(0).normal(size=(2, 512)).astype(np.float32)
    e /= np.linalg.norm(e, axis=1, keepdims=True)
    np.savez(path, names=names, E=e, n_used=np.array([38, 35]))


def test_load_real_embeddings_returns_name_to_vector_dict(tmp_path):
    npz_path = tmp_path / "prototypes.npz"
    _write_fake_npz(npz_path)

    result = load_real_embeddings(npz_path)

    assert set(result.keys()) == {"Tom Hanks", "Meryl Streep"}
    assert result["Tom Hanks"].shape == (512,)
    assert result["Tom Hanks"].dtype == np.float32


def test_load_popularity_returns_name_to_n_used_dict(tmp_path):
    npz_path = tmp_path / "prototypes.npz"
    _write_fake_npz(npz_path)

    result = load_popularity(npz_path)

    assert result == {"Tom Hanks": 38, "Meryl Streep": 35}


def test_filter_prototypes_drops_thin_prototypes_below_min_images():
    embeddings = {
        "Thin Person": np.array([1.0, 0.0]),
        "Solid Person": np.array([0.0, 1.0]),
    }
    popularity = {"Thin Person": 2, "Solid Person": 10}

    kept_embeddings, kept_popularity = filter_prototypes(
        embeddings, popularity, min_images=5, dup_threshold=0.55
    )

    assert set(kept_embeddings.keys()) == {"Solid Person"}
    assert set(kept_popularity.keys()) == {"Solid Person"}


def test_filter_prototypes_drops_near_duplicate_keeping_more_popular():
    # Replicates doppelmap_ipynb.py's dedup step: mean-center + renormalize
    # the surviving embeddings, then for any pair with similarity above the
    # threshold, drop whichever has fewer images (n_used). "Winner" and
    # "Loser" share an identical embedding (e.g. same person crawled under
    # two name spellings); "Distinct" is a different identity entirely.
    embeddings = {
        "Winner": np.array([1.0, 0.0]),
        "Loser": np.array([1.0, 0.0]),
        "Distinct": np.array([0.0, 1.0]),
    }
    popularity = {"Winner": 40, "Loser": 10, "Distinct": 20}

    kept_embeddings, kept_popularity = filter_prototypes(
        embeddings, popularity, min_images=5, dup_threshold=0.55
    )

    assert set(kept_embeddings.keys()) == {"Winner", "Distinct"}
    assert set(kept_popularity.keys()) == {"Winner", "Distinct"}


def test_filter_prototypes_keeps_distinct_people_unchanged():
    embeddings = {
        "Alice": np.array([1.0, 0.0, 0.0]),
        "Bob": np.array([0.0, 1.0, 0.0]),
        "Carol": np.array([0.0, 0.0, 1.0]),
    }
    popularity = {"Alice": 20, "Bob": 25, "Carol": 30}

    kept_embeddings, kept_popularity = filter_prototypes(
        embeddings, popularity, min_images=5, dup_threshold=0.55
    )

    assert set(kept_embeddings.keys()) == {"Alice", "Bob", "Carol"}
    assert kept_popularity == popularity
