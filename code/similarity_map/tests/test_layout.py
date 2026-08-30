import numpy as np
from similarity_map.pipeline.layout import (
    compute_layout,
    normalize_coords,
    refine_layout_with_forceatlas2,
)


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


def test_refine_layout_with_forceatlas2_pulls_connected_nodes_closer():
    # Two nodes are strongly connected but seeded far apart (simulating
    # UMAP placing similar nodes in different regions of the 2D projection
    # -- a real, observed issue: UMAP preserves local neighborhoods in the
    # original high-dim space well, but its 2D projection can still place a
    # directly-connected pair far apart). ForceAtlas2's edge attraction
    # should pull them together much more than an unconnected pair.
    xy = np.array([[0.0, 0.0], [100.0, 100.0], [50.0, 0.0], [50.0, 100.0]])
    edges = [(0, 1, 0.9)]
    initial_dist = np.linalg.norm(xy[0] - xy[1])

    refined = refine_layout_with_forceatlas2(xy, edges, max_iter=500, seed=1)
    refined_dist = np.linalg.norm(refined[0] - refined[1])

    assert refined_dist < initial_dist * 0.1


def test_refine_layout_with_forceatlas2_preserves_shape():
    xy = np.random.default_rng(0).normal(size=(10, 2))
    edges = [(0, 1, 0.5), (2, 3, 0.7)]
    refined = refine_layout_with_forceatlas2(xy, edges, max_iter=50, seed=1)
    assert refined.shape == (10, 2)


def test_refine_layout_with_forceatlas2_handles_no_edges():
    xy = np.random.default_rng(0).normal(size=(5, 2))
    refined = refine_layout_with_forceatlas2(xy, [], max_iter=20, seed=1)
    assert refined.shape == (5, 2)
    assert np.all(np.isfinite(refined))


def test_refine_layout_with_forceatlas2_is_deterministic_for_fixed_positions():
    xy = np.random.default_rng(0).normal(size=(8, 2))
    edges = [(0, 1, 0.5), (2, 3, 0.7), (4, 5, 0.3)]
    a = refine_layout_with_forceatlas2(xy, edges, max_iter=100, seed=7)
    b = refine_layout_with_forceatlas2(xy, edges, max_iter=100, seed=7)
    assert np.allclose(a, b)
