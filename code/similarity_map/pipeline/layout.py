"""2-D layout for the similarity graph: UMAP, a soft pull-in of the extreme
outliers, then normalization to a fixed canvas.
"""
import numpy as np
import umap


def compute_layout(
    embeddings: np.ndarray, *, n_neighbors: int = 15, seed: int = 42
) -> np.ndarray:
    """Returns an (n, 2) float array of raw UMAP coordinates."""
    n = embeddings.shape[0]
    n_neighbors = min(n_neighbors, max(2, n - 1))
    # UMAP's default "spectral" initialization calls an eigensolver that
    # requires k < N and crashes below ~10 points -- a popularity-level
    # subset can plausibly be this small. "random" init avoids it and is
    # still deterministic given random_state.
    init = "random" if n < 10 else "spectral"
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=0.15,
        metric="cosine",
        random_state=seed,
        init=init,
    )
    return reducer.fit_transform(embeddings)


def compress_outliers(
    xy: np.ndarray, *, max_radius_percentile: float = 95.0, tail: float = 0.35
) -> np.ndarray:
    """Softly reins in the furthest-out points: anything beyond the given
    percentile radius keeps its direction from the centroid but has its
    *excess* distance scaled down by `tail`.

    An earlier version pinned outliers to sit exactly on the percentile
    radius. That put 5% of all nodes (66 of 1,330 at the top20 level) at
    one identical distance from the centre -- a visible hard ring of dots
    around the map, which was a large part of why the layout read as "a
    stretched circle". Compressing the tail instead keeps outliers ordered
    relative to each other and leaves no hard edge: measured rim (share of
    nodes within 0.1% of the maximum radius) drops from ~4.9% to ~0.2%,
    while the furthest point still comes in from 1.09x the p95 radius to
    1.03x.
    """
    if xy.shape[0] < 2:
        return xy.copy()
    centroid = xy.mean(axis=0)
    offsets = xy - centroid
    distances = np.linalg.norm(offsets, axis=1)
    max_radius = np.percentile(distances, max_radius_percentile)
    safe_distances = np.where(distances == 0, 1.0, distances)
    compressed = np.where(
        distances <= max_radius, distances, max_radius + (distances - max_radius) * tail
    )
    return centroid + offsets * (compressed / safe_distances)[:, None]


def normalize_coords(xy: np.ndarray, canvas_size: float = 10000.0) -> np.ndarray:
    """Scale/translate coordinates to fill [0, canvas_size]^2, preserving
    aspect ratio.
    """
    mins = xy.min(axis=0)
    maxs = xy.max(axis=0)
    span = maxs - mins
    span = np.where(span == 0, 1.0, span)  # degenerate axis (e.g. n=1)
    scale = canvas_size / span.max()
    return (xy - mins) * scale
