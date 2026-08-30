"""kNN + mutual-kNN similarity graph construction over pre-computed
L2-normalized embeddings.
"""
import numpy as np
from sklearn.neighbors import NearestNeighbors


def build_knn(embeddings: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Cosine kNN over an (n, d) L2-normalized embedding matrix.

    Returns (neighbor_indices, similarities), each shape (n, k), excluding
    self-matches, sorted by descending similarity.
    """
    n = embeddings.shape[0]
    k_query = min(k + 1, n)
    nn = NearestNeighbors(n_neighbors=k_query, metric="cosine").fit(embeddings)
    dist, idx = nn.kneighbors(embeddings)
    sim = 1 - dist
    return idx[:, 1:], sim[:, 1:]


def mutual_knn_edges(
    neighbor_idx: np.ndarray, sim: np.ndarray
) -> list[tuple[int, int, float]]:
    """Keep edge (a, b) only if a is in b's kNN list AND b is in a's kNN
    list. Returns deduplicated undirected edges as (min_id, max_id, weight),
    sorted by (src, dst).
    """
    neighbor_sets = [set(row.tolist()) for row in neighbor_idx]
    edges = {}
    n = neighbor_idx.shape[0]
    for a in range(n):
        for pos, b in enumerate(neighbor_idx[a]):
            b = int(b)
            if a in neighbor_sets[b]:
                key = (min(a, b), max(a, b))
                edges[key] = round(float(sim[a, pos]), 3)
    return sorted((a, b, w) for (a, b), w in edges.items())


def directed_similar_lists(
    neighbor_idx: np.ndarray, sim: np.ndarray
) -> dict[int, list[list]]:
    """Per-node ranked top-k similar list (directed, not mutual-filtered) —
    used for the sidebar, which should show a full top-k even for nodes the
    mutual filter would otherwise isolate.
    """
    result = {}
    for i in range(neighbor_idx.shape[0]):
        result[i] = [[int(j), round(float(s), 3)] for j, s in zip(neighbor_idx[i], sim[i])]
    return result


def mutual_degrees(edges: list[tuple[int, int, float]], n: int) -> list[int]:
    deg = [0] * n
    for a, b, _ in edges:
        deg[a] += 1
        deg[b] += 1
    return deg
