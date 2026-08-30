import numpy as np
from similarity_map.pipeline.layout import (
    clamp_outliers,
    compute_layout,
    normalize_coords,
    refine_layout_with_neighbor_attraction,
)


def test_compute_layout_shape():
    embeddings = np.random.default_rng(0).normal(size=(30, 512)).astype(np.float32)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    xy = compute_layout(embeddings, seed=1)
    assert xy.shape == (30, 2)


def test_compute_layout_handles_very_small_n():
    # UMAP's default "spectral" initialization crashes below ~10 points
    # (its eigensolver requires k < N; discovered when a popularity-level
    # subset happened to be this small in a test fixture). A real dataset
    # could plausibly hit a small level too, so this must not crash.
    embeddings = np.random.default_rng(0).normal(size=(3, 512)).astype(np.float32)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    xy = compute_layout(embeddings, seed=1)
    assert xy.shape == (3, 2)
    assert np.all(np.isfinite(xy))


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
    # (Real-data investigation: tried adding gravity here to fix far-flung
    # isolated-node outliers, but it doesn't work -- gravity shrinks the
    # attraction-connected cluster just as fast as it shrinks isolated
    # nodes, so the *relative* outlier problem survives unchanged. See
    # clamp_outliers below for the fix that actually worked.)
    xy = np.array([[0.0, 0.0], [10.0, 10.0], [500.0, 500.0], [-500.0, 500.0]])
    edges = [(0, 1, 0.8)]

    refined = refine_layout_with_neighbor_attraction(xy, edges, iterations=100)

    assert np.allclose(refined[2], xy[2])
    assert np.allclose(refined[3], xy[3])


def test_clamp_outliers_pulls_in_points_beyond_the_percentile_radius():
    # Real data showed this exact problem: isolated (degree-0) nodes had a
    # median distance-from-centroid more than double the connected nodes'.
    # Rather than fight that with competing forces, directly cap how far
    # from the centroid any point can end up: anything beyond the given
    # percentile radius gets pulled straight in to sit exactly on it,
    # preserving its direction from the centroid.
    #
    # 20 inlier points on a small circle, plus one clear outlier -- enough
    # inliers that the percentile threshold reflects their spread rather
    # than being dragged up by the single outlier itself (this is what the
    # real 6,537-node dataset looked like: ~5% genuine outliers against a
    # large inlier population).
    rng = np.random.default_rng(0)
    angles = np.linspace(0, 2 * np.pi, 20, endpoint=False)
    inliers = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    xy = np.vstack([inliers, [[1000.0, 1000.0]]])

    clamped = clamp_outliers(xy, max_radius_percentile=90.0)

    centroid = xy.mean(axis=0)
    dist_before = np.linalg.norm(xy[-1] - centroid)
    dist_after = np.linalg.norm(clamped[-1] - centroid)
    assert dist_after < dist_before * 0.1
    # the direction from centroid must be preserved, only magnitude capped
    direction_before = (xy[-1] - centroid) / dist_before
    direction_after = (clamped[-1] - centroid) / dist_after
    assert np.allclose(direction_before, direction_after, atol=1e-6)


def test_clamp_outliers_leaves_points_within_radius_untouched():
    xy = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    clamped = clamp_outliers(xy, max_radius_percentile=95.0)
    assert np.allclose(clamped, xy)


def test_clamp_outliers_preserves_shape_and_handles_degenerate_input():
    xy = np.array([[5.0, 5.0]])
    clamped = clamp_outliers(xy)
    assert clamped.shape == (1, 2)
    assert np.all(np.isfinite(clamped))


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
