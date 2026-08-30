"""CLI: assemble one graph-<level>.json + thumbs/ per popularity level,
matching the doppelmap frontend's schema, from real ArcFace embeddings
loaded from a .npz produced by the notebook pipeline.

Real profile photos are NOT fetched here -- the frontend looks them up live
from Wikipedia when a node is clicked, so dataset generation stays fully
offline regardless of dataset size. Every node gets a locally-generated
placeholder avatar as the thumbnail shown before (or in place of) a real
photo.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .real_embeddings import filter_prototypes, load_popularity, load_real_embeddings
from .graph import build_knn, mutual_knn_edges, directed_similar_lists, mutual_degrees
from .layout import (
    clamp_outliers,
    compute_layout,
    normalize_coords,
    refine_layout_with_neighbor_attraction,
)
from .thumbnails import generate_thumbnail

DEFAULT_OUT = Path(__file__).resolve().parents[2] / "web" / "public" / "data"

# Filtering to a popularity level gets its own standalone kNN graph + layout
# (computed only among that level's members) rather than just hiding nodes
# in one big layout -- the latter left the visible subset scattered
# relative to positions computed against a much larger graph. Percentiles
# are of the *popularity* distribution (n_used), matching the frontend's
# 4-level slider exactly.
POPULARITY_LEVELS: dict[str, float] = {
    "all": 0.0,
    "top50": 50.0,
    "top20": 80.0,
    "top5": 95.0,
}


def _thumb_filename(name: str) -> str:
    # Hash-based rather than name-derived: filename-safe regardless of
    # accents/apostrophes/etc in real names, and -- the actual point here --
    # stable across every level's file, so a person appearing in multiple
    # levels shares one thumbnail instead of getting a duplicate per level.
    digest = hashlib.sha1(name.encode("utf-8")).hexdigest()[:16]
    return f"{digest}.webp"


def _assemble_graph(
    embeddings_by_name: dict[str, np.ndarray],
    *,
    k: int,
    seed: int,
    out_dir: Path,
    version: str,
    default_attr: str,
    popularity_by_name: dict[str, int] | None = None,
    graph_filename: str = "graph.json",
    similar_k: int = 15,
) -> dict:
    names = list(embeddings_by_name.keys())
    embeddings = np.stack([embeddings_by_name[n] for n in names])

    # The sidebar's ranked "similar" list wants more entries (similar_k)
    # than the visual mutual-kNN graph should have edges (k) -- fetch the
    # wider neighbor list once and slice both from it, rather than two
    # separate kNN passes. build_knn already sorts each row by descending
    # similarity, so the first k columns are exactly the same top-k that a
    # k-only call would have returned.
    neighbor_idx, sim = build_knn(embeddings, k=max(k, similar_k))
    edges = mutual_knn_edges(neighbor_idx[:, :k], sim[:, :k])
    similar = directed_similar_lists(neighbor_idx[:, :similar_k], sim[:, :similar_k])
    deg = mutual_degrees(edges, n=len(names))

    umap_xy = compute_layout(embeddings, seed=seed)
    refined_xy = refine_layout_with_neighbor_attraction(umap_xy, edges)
    refined_xy = clamp_outliers(refined_xy)
    xy = normalize_coords(refined_xy)

    out_dir.mkdir(parents=True, exist_ok=True)
    thumbs_dir = out_dir / "thumbs"
    thumbs_dir.mkdir(parents=True, exist_ok=True)
    nodes = []
    for i, name in enumerate(names):
        thumb_filename = _thumb_filename(name)
        thumb_path = thumbs_dir / thumb_filename
        if not thumb_path.exists():
            generate_thumbnail(name, thumb_path)
        nodes.append({
            "id": i,
            "name": name,
            "x": round(float(xy[i, 0]), 1),
            "y": round(float(xy[i, 1]), 1),
            "deg": deg[i],
            "thumb": f"thumbs/{thumb_filename}",
            "attr": default_attr,
            "popularity": popularity_by_name[name] if popularity_by_name else 1,
        })

    graph = {
        "meta": {"version": version, "count": len(names), "k": k},
        "nodes": nodes,
        "edges": [[a, b, w] for a, b, w in edges],
        "similar": {str(i): ranked for i, ranked in similar.items()},
    }

    (out_dir / graph_filename).write_text(json.dumps(graph), encoding="utf-8")
    return graph


def build_dataset_from_embeddings(
    embeddings_path: Path,
    *,
    k: int,
    seed: int,
    out_dir: Path,
    levels: dict[str, float] = POPULARITY_LEVELS,
) -> dict[str, dict]:
    """Real dataset: load precomputed ArcFace embeddings from a .npz (see
    real_embeddings.load_real_embeddings), drop thin/duplicate prototypes
    (real_embeddings.filter_prototypes), then build one standalone graph per
    popularity level -- each with its own kNN graph, layout, and outlier
    clamp computed only among that level's members (not a filtered view of
    one big graph, which left the visible subset scattered relative to
    positions computed against far more people than were actually shown).
    Returns {level_label: graph}, one entry per `levels`.
    """
    embeddings_by_name = load_real_embeddings(embeddings_path)
    popularity_by_name = load_popularity(embeddings_path)
    embeddings_by_name, popularity_by_name = filter_prototypes(
        embeddings_by_name, popularity_by_name
    )

    all_popularities = np.array(list(popularity_by_name.values()))
    graphs: dict[str, dict] = {}
    for label, percentile in levels.items():
        threshold = (
            np.percentile(all_popularities, percentile)
            if percentile > 0
            else all_popularities.min()
        )
        level_names = [n for n in embeddings_by_name if popularity_by_name[n] >= threshold]
        level_embeddings = {n: embeddings_by_name[n] for n in level_names}
        level_popularity = {n: popularity_by_name[n] for n in level_names}
        graphs[label] = _assemble_graph(
            level_embeddings,
            k=k,
            seed=seed,
            out_dir=out_dir,
            version=f"real-{embeddings_path.stem}-{label}",
            default_attr="Placeholder avatar — real photo not yet linked",
            popularity_by_name=level_popularity,
            graph_filename=f"graph-{label}.json",
        )
    return graphs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--embeddings", type=Path, required=True,
        help="path to a real-embeddings .npz (names + E arrays), using every name in the file",
    )
    parser.add_argument("--k", type=int, default=8, help="kNN neighbors per node")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    graphs = build_dataset_from_embeddings(
        args.embeddings, k=args.k, seed=args.seed, out_dir=args.out
    )
    for label, graph in graphs.items():
        print(
            f"[{label}] Wrote {len(graph['nodes'])} nodes, {len(graph['edges'])} edges "
            f"to {args.out}/graph-{label}.json"
        )


if __name__ == "__main__":
    main()
