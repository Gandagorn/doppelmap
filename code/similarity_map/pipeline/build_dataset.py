"""CLI: assemble graph.json + thumbs/ matching the doppelmap frontend's
schema (doppelmap-lld.md section 2.5), from either synthetic cluster-
structured embeddings (default, for prototyping) or real ArcFace embeddings
loaded from a .npz produced by the notebook pipeline (--embeddings).

Real profile photos are NOT fetched here -- the frontend looks them up live
from Wikipedia when a node is clicked, so dataset generation stays fully
offline regardless of dataset size. Every node gets a locally-generated
placeholder avatar as the thumbnail shown before (or in place of) a real
photo.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from .names import CELEBRITY_NAMES
from .embeddings import generate_synthetic_embeddings
from .real_embeddings import filter_prototypes, load_popularity, load_real_embeddings
from .graph import build_knn, mutual_knn_edges, directed_similar_lists, mutual_degrees
from .layout import compute_layout, normalize_coords, refine_layout_with_forceatlas2
from .thumbnails import generate_thumbnail

DEFAULT_OUT = Path(__file__).resolve().parents[2] / "web" / "public" / "data"


def _assemble_graph(
    embeddings_by_name: dict[str, np.ndarray],
    *,
    k: int,
    seed: int,
    out_dir: Path,
    version: str,
    default_attr: str,
    popularity_by_name: dict[str, int] | None = None,
) -> dict:
    names = list(embeddings_by_name.keys())
    embeddings = np.stack([embeddings_by_name[n] for n in names])

    neighbor_idx, sim = build_knn(embeddings, k=k)
    edges = mutual_knn_edges(neighbor_idx, sim)
    similar = directed_similar_lists(neighbor_idx, sim)
    deg = mutual_degrees(edges, n=len(names))

    umap_xy = compute_layout(embeddings, seed=seed)
    refined_xy = refine_layout_with_forceatlas2(umap_xy, edges, seed=seed)
    xy = normalize_coords(refined_xy)

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
            "attr": default_attr,
            "popularity": popularity_by_name[name] if popularity_by_name else 1,
        })

    graph = {
        "meta": {"version": version, "count": len(names), "k": k},
        "nodes": nodes,
        "edges": [[a, b, w] for a, b, w in edges],
        "similar": {str(i): ranked for i, ranked in similar.items()},
    }

    (out_dir / "graph.json").write_text(json.dumps(graph), encoding="utf-8")
    return graph


def build_dataset(*, count: int, k: int, seed: int, out_dir: Path) -> dict:
    """Synthetic dataset: curated name list + generated cluster-structured
    embeddings, for prototyping before real embeddings exist.
    """
    if count <= 0:
        raise ValueError(f"count must be positive, got {count}")
    names = CELEBRITY_NAMES[:count]
    if len(names) < count:
        raise ValueError(
            f"requested count={count} but only {len(names)} curated names are available"
        )

    embeddings_by_name = generate_synthetic_embeddings(names, seed=seed)
    return _assemble_graph(
        embeddings_by_name,
        k=k,
        seed=seed,
        out_dir=out_dir,
        version="synthetic-2026-08-29",
        default_attr="Synthetic placeholder — no real photo",
    )


def build_dataset_from_embeddings(embeddings_path: Path, *, k: int, seed: int, out_dir: Path) -> dict:
    """Real dataset: load precomputed ArcFace embeddings from a .npz (see
    real_embeddings.load_real_embeddings), drop thin/duplicate prototypes
    (real_embeddings.filter_prototypes), then use every remaining name --
    no count cap.
    """
    embeddings_by_name = load_real_embeddings(embeddings_path)
    popularity_by_name = load_popularity(embeddings_path)
    embeddings_by_name, popularity_by_name = filter_prototypes(
        embeddings_by_name, popularity_by_name
    )
    return _assemble_graph(
        embeddings_by_name,
        k=k,
        seed=seed,
        out_dir=out_dir,
        version=f"real-{embeddings_path.stem}",
        default_attr="Placeholder avatar — real photo not yet linked",
        popularity_by_name=popularity_by_name,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--count", type=int, default=150,
        help="number of synthetic celebrities (ignored when --embeddings is given)",
    )
    parser.add_argument("--k", type=int, default=8, help="kNN neighbors per node")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--embeddings", type=Path, default=None,
        help="path to a real-embeddings .npz (names + E arrays); when given, "
        "builds from real data instead of synthetic, using every name in the file",
    )
    args = parser.parse_args()
    if args.embeddings:
        graph = build_dataset_from_embeddings(
            args.embeddings, k=args.k, seed=args.seed, out_dir=args.out
        )
    else:
        graph = build_dataset(count=args.count, k=args.k, seed=args.seed, out_dir=args.out)
    print(f"Wrote {len(graph['nodes'])} nodes, {len(graph['edges'])} edges to {args.out}")


if __name__ == "__main__":
    main()
