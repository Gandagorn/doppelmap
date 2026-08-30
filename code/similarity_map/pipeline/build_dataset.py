"""CLI: generate a fully synthetic graph.json + thumbs/ matching the real
doppelmap pipeline's output schema (doppelmap-lld.md section 2.5), using
synthetic embeddings instead of real face embeddings so the frontend can be
built and tested before the real pipeline exists.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from .names import CELEBRITY_NAMES
from .embeddings import generate_synthetic_embeddings
from .graph import build_knn, mutual_knn_edges, directed_similar_lists, mutual_degrees
from .layout import compute_layout, normalize_coords
from .thumbnails import generate_thumbnail

DEFAULT_OUT = Path(__file__).resolve().parents[2] / "web" / "public" / "data"


def build_dataset(*, count: int, k: int, seed: int, out_dir: Path) -> dict:
    if count <= 0:
        raise ValueError(f"count must be positive, got {count}")
    names = CELEBRITY_NAMES[:count]
    if len(names) < count:
        raise ValueError(
            f"requested count={count} but only {len(names)} curated names are available"
        )

    embeddings_by_name = generate_synthetic_embeddings(names, seed=seed)
    embeddings = np.stack([embeddings_by_name[n] for n in names])

    neighbor_idx, sim = build_knn(embeddings, k=k)
    edges = mutual_knn_edges(neighbor_idx, sim)
    similar = directed_similar_lists(neighbor_idx, sim)
    deg = mutual_degrees(edges, n=len(names))

    xy = normalize_coords(compute_layout(embeddings, seed=seed))

    out_dir.mkdir(parents=True, exist_ok=True)
    thumbs_dir = out_dir / "thumbs"
    nodes = []
    for i, name in enumerate(names):
        thumb_rel = f"thumbs/{i}.webp"
        generate_thumbnail(name, thumbs_dir / f"{i}.webp")
        nodes.append({
            "id": i,
            "name": name,
            "x": round(float(xy[i, 0]), 1),
            "y": round(float(xy[i, 1]), 1),
            "deg": deg[i],
            "thumb": thumb_rel,
            "attr": "Synthetic placeholder — no real photo",
        })

    graph = {
        "meta": {"version": "synthetic-2026-08-29", "count": len(names), "k": k},
        "nodes": nodes,
        "edges": [[a, b, w] for a, b, w in edges],
        "similar": {str(i): ranked for i, ranked in similar.items()},
    }

    (out_dir / "graph.json").write_text(json.dumps(graph), encoding="utf-8")
    return graph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=150, help="number of synthetic celebrities")
    parser.add_argument("--k", type=int, default=8, help="kNN neighbors per node")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    graph = build_dataset(count=args.count, k=args.k, seed=args.seed, out_dir=args.out)
    print(f"Wrote {len(graph['nodes'])} nodes, {len(graph['edges'])} edges to {args.out}")


if __name__ == "__main__":
    main()
