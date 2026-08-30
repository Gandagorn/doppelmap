"""2-D layout for the similarity graph: UMAP for global structure, refined
with ForceAtlas2 on the mutual-kNN graph so edge length reflects graph
distance, normalized to a fixed canvas.
"""
import networkx as nx
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


def refine_layout_with_forceatlas2(
    xy: np.ndarray,
    edges: list[tuple[int, int, float]],
    *,
    max_iter: int = 500,
    seed: int = 42,
) -> np.ndarray:
    """Refines a UMAP layout with ForceAtlas2 on the mutual-kNN graph, seeded
    from the UMAP positions. UMAP preserves local neighborhoods well in the
    original high-dimensional space, but its 2D projection can still place a
    directly-connected (mutual-kNN) pair far apart -- ForceAtlas2's edge
    attraction (weighted by similarity) pulls connected nodes together and
    its gravity keeps disconnected nodes from drifting away, so the final
    2D distance actually reflects graph distance.
    """
    n = xy.shape[0]
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    for a, b, w in edges:
        graph.add_edge(a, b, weight=w)

    # forceatlas2_layout requires numpy-array position values (plain tuples
    # raise AttributeError on the internal `.copy()` call).
    initial_pos = {i: np.array([xy[i, 0], xy[i, 1]], dtype=np.float64) for i in range(n)}
    refined = nx.forceatlas2_layout(
        graph, pos=initial_pos, max_iter=max_iter, weight="weight", seed=seed
    )
    return np.array([refined[i] for i in range(n)], dtype=np.float64)


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
