"""2-D layout for the similarity graph: UMAP for global structure, refined
with sparse neighbor-attraction on the mutual-kNN graph so edge length
reflects graph distance, normalized to a fixed canvas.
"""
import numpy as np
import umap
from scipy.sparse import csr_matrix


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


def refine_layout_with_neighbor_attraction(
    xy: np.ndarray,
    edges: list[tuple[int, int, float]],
    *,
    iterations: int = 100,
    attraction_strength: float = 0.2,
) -> np.ndarray:
    """Refines a UMAP layout by repeatedly nudging each node toward the
    similarity-weighted centroid of its mutual-kNN neighbors. UMAP preserves
    local neighborhoods well in the original high-dimensional space, but its
    2D projection can still place a directly-connected pair far apart --
    this pulls connected nodes together so 2D distance actually reflects
    graph distance. Nodes with no edges are left exactly where UMAP put them.

    An earlier version used ForceAtlas2 (all-pairs repulsion each
    iteration), which is O(n^2) per iteration -- fine at a few hundred
    nodes, but ~37 minutes extrapolated at 10,000 (measured 203s at 3,000).
    This is O(iterations * edges) via a sparse adjacency matvec each step,
    so it scales to real dataset sizes.
    """
    n = xy.shape[0]
    if not edges:
        return xy.copy()

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    for a, b, w in edges:
        rows += [a, b]
        cols += [b, a]
        vals += [w, w]
    adjacency = csr_matrix((vals, (rows, cols)), shape=(n, n))
    degree = np.asarray(adjacency.sum(axis=1)).flatten()
    has_neighbors = degree > 0
    safe_degree = np.where(has_neighbors, degree, 1.0)

    pos = xy.astype(np.float64).copy()
    for _ in range(iterations):
        centroid = (adjacency @ pos) / safe_degree[:, None]
        delta = attraction_strength * (centroid - pos)
        delta[~has_neighbors] = 0.0
        pos = pos + delta
    return pos


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
