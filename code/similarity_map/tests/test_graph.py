import numpy as np
from similarity_map.pipeline.graph import (
    build_knn,
    mutual_knn_edges,
    directed_similar_lists,
    mutual_degrees,
)


def _two_tight_clusters():
    cluster_a = np.array([
        [1.0, 0.0, 0.0],
        [0.99, 0.01, 0.0],
        [0.98, 0.0, 0.02],
    ])
    cluster_b = np.array([
        [0.0, 1.0, 0.0],
        [0.01, 0.99, 0.0],
        [0.0, 0.98, 0.02],
    ])
    embeddings = np.vstack([cluster_a, cluster_b])
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings.astype(np.float32)


def test_build_knn_excludes_self_and_sorts_descending():
    embeddings = _two_tight_clusters()
    idx, sim = build_knn(embeddings, k=2)
    assert idx.shape == (6, 2)
    for row in range(6):
        assert row not in idx[row].tolist()
        assert sim[row, 0] >= sim[row, 1]


def test_mutual_knn_only_connects_within_cluster():
    embeddings = _two_tight_clusters()
    idx, sim = build_knn(embeddings, k=2)
    edges = mutual_knn_edges(idx, sim)
    assert len(edges) > 0
    for a, b, w in edges:
        assert (a < 3) == (b < 3)
        assert 0.0 <= w <= 1.0


def test_directed_similar_lists_ranked_and_covers_every_node():
    embeddings = _two_tight_clusters()
    idx, sim = build_knn(embeddings, k=2)
    similar = directed_similar_lists(idx, sim)
    assert set(similar.keys()) == set(range(6))
    for ranked in similar.values():
        sims = [s for _, s in ranked]
        assert sims == sorted(sims, reverse=True)


def test_mutual_degrees_matches_edge_count():
    embeddings = _two_tight_clusters()
    idx, sim = build_knn(embeddings, k=2)
    edges = mutual_knn_edges(idx, sim)
    deg = mutual_degrees(edges, n=6)
    assert sum(deg) == 2 * len(edges)
