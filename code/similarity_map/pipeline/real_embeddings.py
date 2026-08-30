"""Loads real ArcFace embeddings from a .npz produced by the notebook
pipeline described in doppelmap-lld.md: `names` (str array), `E`
((n, 512) float32, L2-normalized) and `n_used` (int, images kept per
person) arrays, one row per name.
"""
from pathlib import Path

import numpy as np

# Mirrors doppelmap_ipynb.py's post-save cleanup (MIN_IMAGES / DUP), which
# in the notebook runs *after* its own np.savez call -- so the raw .npz
# doesn't reflect it yet. Replicated here so our pipeline is correct
# regardless of which notebook revision produced the file.
MIN_IMAGES = 5
DUP_SIMILARITY = 0.55


def load_real_embeddings(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    names = data["names"].tolist()
    embeddings = data["E"]
    return {name: embeddings[i].astype(np.float32) for i, name in enumerate(names)}


def load_popularity(path: Path) -> dict[str, int]:
    data = np.load(path, allow_pickle=True)
    names = data["names"].tolist()
    n_used = data["n_used"]
    return {name: int(n_used[i]) for i, name in enumerate(names)}


def filter_prototypes(
    embeddings_by_name: dict[str, np.ndarray],
    popularity_by_name: dict[str, int],
    *,
    min_images: int = MIN_IMAGES,
    dup_threshold: float = DUP_SIMILARITY,
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """Drops thin prototypes (fewer than `min_images` source photos), then
    drops near-duplicate name entries -- e.g. the same person crawled under
    two name spellings -- keeping whichever has more images. Duplicate
    detection mean-centers and renormalizes the surviving embeddings first
    (this is what actually separates "same identity" from "generic face
    similarity" -- raw ArcFace embeddings all point in a similar general
    direction), exactly matching the notebook's own dedup step.
    """
    names = [n for n in embeddings_by_name if popularity_by_name.get(n, 0) >= min_images]
    if len(names) < 2:
        return (
            {n: embeddings_by_name[n] for n in names},
            {n: popularity_by_name[n] for n in names},
        )

    embeddings = np.stack([embeddings_by_name[n] for n in names])
    centered = embeddings - embeddings.mean(axis=0)
    centered /= np.linalg.norm(centered, axis=1, keepdims=True)
    similarity = centered @ centered.T

    dup_a, dup_b = np.where(np.triu(similarity, k=1) > dup_threshold)
    drop = set()
    for a, b in zip(dup_a.tolist(), dup_b.tolist()):
        loser = a if popularity_by_name[names[a]] < popularity_by_name[names[b]] else b
        drop.add(loser)

    kept = [names[i] for i in range(len(names)) if i not in drop]
    return (
        {name: embeddings_by_name[name] for name in kept},
        {name: popularity_by_name[name] for name in kept},
    )
