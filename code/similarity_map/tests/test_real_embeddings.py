import numpy as np

from similarity_map.pipeline.real_embeddings import load_real_embeddings


def test_load_real_embeddings_returns_name_to_vector_dict(tmp_path):
    npz_path = tmp_path / "prototypes.npz"
    names = np.array(["Tom Hanks", "Meryl Streep"])
    e = np.random.default_rng(0).normal(size=(2, 512)).astype(np.float32)
    e /= np.linalg.norm(e, axis=1, keepdims=True)
    np.savez(npz_path, names=names, E=e, n_used=np.array([38, 35]))

    result = load_real_embeddings(npz_path)

    assert set(result.keys()) == {"Tom Hanks", "Meryl Streep"}
    assert result["Tom Hanks"].shape == (512,)
    assert result["Tom Hanks"].dtype == np.float32
    assert np.allclose(result["Tom Hanks"], e[0])
    assert np.allclose(result["Meryl Streep"], e[1])
