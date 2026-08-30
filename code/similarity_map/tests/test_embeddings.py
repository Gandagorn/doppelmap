import numpy as np
from similarity_map.pipeline.embeddings import generate_synthetic_embeddings, EMBEDDING_DIM


def test_embeddings_are_unit_normalized():
    names = [f"Person {i}" for i in range(20)]
    embeddings = generate_synthetic_embeddings(names, seed=1)
    for vec in embeddings.values():
        assert vec.shape == (EMBEDDING_DIM,)
        assert abs(np.linalg.norm(vec) - 1.0) < 1e-5


def test_embeddings_deterministic_for_fixed_seed():
    names = [f"Person {i}" for i in range(20)]
    a = generate_synthetic_embeddings(names, seed=7)
    b = generate_synthetic_embeddings(names, seed=7)
    for name in names:
        assert np.allclose(a[name], b[name])


def test_cluster_structure_creates_higher_within_cluster_similarity():
    names = [f"Person {i}" for i in range(60)]
    embeddings = generate_synthetic_embeddings(
        names, avg_cluster_size=6, noise_scale=0.2, seed=3
    )
    vecs = np.stack([embeddings[n] for n in names])
    sim = vecs @ vecs.T
    off_diag = sim[~np.eye(len(names), dtype=bool)]
    best_neighbor_sim = np.sort(sim, axis=1)[:, -2]  # exclude self (always 1.0)
    assert best_neighbor_sim.mean() > off_diag.mean() + 0.15
