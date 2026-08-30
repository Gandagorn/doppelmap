"""Synthetic 512-d ArcFace-shaped embeddings with cluster structure, for
prototyping the pipeline before real face embeddings exist.
"""
import numpy as np

EMBEDDING_DIM = 512


def generate_synthetic_embeddings(
    names: list[str],
    *,
    avg_cluster_size: float = 8.0,
    noise_scale: float = 0.35,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Assign each name to a random synthetic cluster, then draw a unit
    embedding near that cluster's center. Clusters are arbitrary (not tied
    to profession, gender, etc.) — they exist purely so the kNN graph has
    believable structure instead of pure noise.

    Returns name -> L2-normalized float32 vector of shape (EMBEDDING_DIM,).
    """
    rng = np.random.default_rng(seed)
    n = len(names)
    n_clusters = max(1, round(n / avg_cluster_size))
    cluster_centers = rng.normal(size=(n_clusters, EMBEDDING_DIM)).astype(np.float32)
    cluster_centers /= np.linalg.norm(cluster_centers, axis=1, keepdims=True)

    cluster_ids = rng.integers(0, n_clusters, size=n)

    embeddings = {}
    for name, cluster_id in zip(names, cluster_ids):
        center = cluster_centers[cluster_id]
        noise = rng.normal(size=EMBEDDING_DIM).astype(np.float32)
        noise /= np.linalg.norm(noise) + 1e-8
        noise = noise * noise_scale
        vec = center + noise
        vec /= np.linalg.norm(vec)
        embeddings[name] = vec.astype(np.float32)
    return embeddings
