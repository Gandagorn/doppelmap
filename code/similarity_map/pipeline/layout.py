"""2-D layout for the similarity graph: UMAP for global structure,
normalized to a fixed canvas.
"""
import numpy as np
import umap


def compute_layout(
    embeddings: np.ndarray, *, n_neighbors: int = 15, seed: int = 42
) -> np.ndarray:
    """Returns an (n, 2) float array of raw UMAP coordinates."""
    n_neighbors = min(n_neighbors, max(2, embeddings.shape[0] - 1))
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=0.15,
        metric="cosine",
        random_state=seed,
    )
    return reducer.fit_transform(embeddings)


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
