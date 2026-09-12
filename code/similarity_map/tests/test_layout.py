import numpy as np
from similarity_map.pipeline.layout import (
    compress_outliers,
    compute_layout,
    normalize_coords,
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


def _ring_with_one_outlier():
    # 20 inliers on a small circle plus one clear outlier -- enough inliers
    # that the percentile threshold reflects their spread instead of being
    # dragged up by the outlier itself, mirroring the real datasets (~5%
    # genuine outliers against a large inlier population).
    angles = np.linspace(0, 2 * np.pi, 20, endpoint=False)
    inliers = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    return np.vstack([inliers, [[1000.0, 1000.0]]])


def test_compress_outliers_pulls_in_points_beyond_the_percentile_radius():
    xy = _ring_with_one_outlier()

    compressed = compress_outliers(xy, max_radius_percentile=90.0)

    centroid = xy.mean(axis=0)
    dist_before = np.linalg.norm(xy[-1] - centroid)
    dist_after = np.linalg.norm(compressed[-1] - centroid)
    assert dist_after < dist_before * 0.5
    # the direction from centroid must be preserved, only magnitude reduced
    direction_before = (xy[-1] - centroid) / dist_before
    direction_after = (compressed[-1] - centroid) / dist_after
    assert np.allclose(direction_before, direction_after, atol=1e-6)


def test_compress_outliers_keeps_outliers_ordered_instead_of_pinning_them():
    # The behaviour that matters visually. Pinning every outlier to exactly
    # the percentile radius put ~5% of nodes at one identical distance from
    # the centre, drawing a hard ring around the map. Compressing the tail
    # must leave a further-out point still further out than a nearer one.
    angles = np.linspace(0, 2 * np.pi, 20, endpoint=False)
    inliers = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    xy = np.vstack([inliers, [[50.0, 0.0]], [[500.0, 0.0]]])

    compressed = compress_outliers(xy, max_radius_percentile=90.0)

    centroid = xy.mean(axis=0)
    near = np.linalg.norm(compressed[-2] - centroid)
    far = np.linalg.norm(compressed[-1] - centroid)
    assert far > near * 1.5


def test_compress_outliers_leaves_points_within_radius_untouched():
    xy = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    compressed = compress_outliers(xy, max_radius_percentile=95.0)
    assert np.allclose(compressed, xy)


def test_compress_outliers_preserves_shape_and_handles_degenerate_input():
    xy = np.array([[5.0, 5.0]])
    compressed = compress_outliers(xy)
    assert compressed.shape == (1, 2)
    assert np.all(np.isfinite(compressed))
