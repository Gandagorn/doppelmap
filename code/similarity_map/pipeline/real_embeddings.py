"""Loads real ArcFace embeddings from a .npz produced by the notebook
pipeline described in doppelmap-lld.md: `names` (str array) and `E`
((n, 512) float32, L2-normalized) arrays, one row of E per name.
"""
from pathlib import Path

import numpy as np


def load_real_embeddings(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    names = data["names"].tolist()
    embeddings = data["E"]
    return {name: embeddings[i].astype(np.float32) for i, name in enumerate(names)}
