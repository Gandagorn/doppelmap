import numpy as np
from similarity_map.pipeline.layout import (
    compute_layout,
    normalize_coords,
    refine_layout_with_neighbor_attraction,
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


def test_refine_layout_with_neighbor_attraction_pulls_connected_nodes_closer():
    # Two nodes are strongly connected but seeded far apart (simulating
    # UMAP placing similar nodes in different regions of the 2D projection).
    # Repeatedly nudging each node toward its (similarity-weighted) neighbor
    # centroid should pull them together much more than an unconnected pair.
    xy = np.array([[0.0, 0.0], [100.0, 100.0], [50.0, 0.0], [50.0, 100.0]])
    edges = [(0, 1, 0.9)]
    initial_dist = np.linalg.norm(xy[0] - xy[1])

    refined = refine_layout_with_neighbor_attraction(xy, edges, iterations=200)
    refined_dist = np.linalg.norm(refined[0] - refined[1])

    assert refined_dist < initial_dist * 0.1


def test_refine_layout_with_neighbor_attraction_preserves_shape():
    xy = np.random.default_rng(0).normal(size=(10, 2))
    edges = [(0, 1, 0.5), (2, 3, 0.7)]
    refined = refine_layout_with_neighbor_attraction(xy, edges, iterations=50)
    assert refined.shape == (10, 2)


def test_refine_layout_with_neighbor_attraction_handles_no_edges():
    # With no edges there is nothing to attract toward -- positions must be
    # left exactly as given, not drift toward the origin or collapse.
    xy = np.random.default_rng(0).normal(size=(5, 2))
    refined = refine_layout_with_neighbor_attraction(xy, [], iterations=20)
    assert np.allclose(refined, xy)


def test_refine_layout_with_neighbor_attraction_leaves_isolated_nodes_in_place():
    # Nodes 2 and 3 have no edges at all, in a graph where 0-1 are
    # connected. They must not drift toward the origin or toward the
    # connected pair -- only nodes with at least one edge should move.
    xy = np.array([[0.0, 0.0], [10.0, 10.0], [500.0, 500.0], [-500.0, 500.0]])
    edges = [(0, 1, 0.8)]

    refined = refine_layout_with_neighbor_attraction(xy, edges, iterations=100)

    assert np.allclose(refined[2], xy[2])
    assert np.allclose(refined[3], xy[3])


def test_refine_layout_with_neighbor_attraction_is_deterministic():
    xy = np.random.default_rng(0).normal(size=(8, 2))
    edges = [(0, 1, 0.5), (2, 3, 0.7), (4, 5, 0.3)]
    a = refine_layout_with_neighbor_attraction(xy, edges, iterations=100)
    b = refine_layout_with_neighbor_attraction(xy, edges, iterations=100)
    assert np.allclose(a, b)


def test_refine_layout_with_neighbor_attraction_scales_to_large_graphs():
    # Regression guard: an earlier ForceAtlas2-based implementation was
    # O(n^2) per iteration and took an estimated ~37 minutes at n=10,000
    # (measured: 203s for n=3,000). This implementation must stay fast at
    # that scale since real datasets have grown to ~10,000 people.
    import time

    rng = np.random.default_rng(0)
    n = 10_000
    xy = rng.normal(size=(n, 2)) * 1000
    edges = [(int(rng.integers(0, n)), int(rng.integers(0, n)), 0.5) for _ in range(n * 8)]

    start = time.time()
    refine_layout_with_neighbor_attraction(xy, edges, iterations=100)
    elapsed = time.time() - start

    assert elapsed < 15.0
