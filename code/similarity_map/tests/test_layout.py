import numpy as np
from similarity_map.pipeline.layout import compute_layout, normalize_coords


def test_compute_layout_shape():
    embeddings = np.random.default_rng(0).normal(size=(30, 512)).astype(np.float32)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    xy = compute_layout(embeddings, seed=1)
    assert xy.shape == (30, 2)


def test_normalize_coords_within_bounds():
    xy = np.array([[-5.0, 100.0], [20.0, -30.0], [0.0, 0.0]])
    normalized = normalize_coords(xy, canvas_size=10000.0)
    assert normalized.min() >= 0.0
    assert normalized.max() <= 10000.0
    assert np.isclose(normalized.max(), 10000.0)


def test_normalize_coords_handles_degenerate_single_point():
    xy = np.array([[3.0, 3.0]])
    normalized = normalize_coords(xy)
    assert normalized.shape == (1, 2)
    assert np.all(np.isfinite(normalized))
