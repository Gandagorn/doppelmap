"""CLI: assemble the site's dataset, matching the frontend's schema.

Two sources. --commons reads the per-person JSON the Wikimedia Commons
collector writes and produces a single graph.json plus photos.json; this
is the current pipeline. --embeddings reads the older IMDB-WIKI .npz and
produces the four popularity-level files; kept so existing data still
builds, but superseded.

The Commons path carries real face crops -- a Commons filename and the
face's box within it -- so the site can show the actual faces rather than
generated placeholder avatars, and can show which two photographs
actually match when two people are compared.
"""
import argparse
import hashlib
import html
import json
import re
from pathlib import Path

import numpy as np

from .real_embeddings import filter_prototypes, load_popularity, load_real_embeddings
from .graph import build_knn, mutual_knn_edges, directed_similar_lists, mutual_degrees
from .layout import compress_outliers, compute_layout, normalize_coords
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

    # UMAP's own output is used directly. An attraction pass used to run
    # here to pull connected nodes together, but measured against the real
    # data it did the opposite of its purpose: it collapsed everything into
    # a dense core, and the share of a node's graph neighbours that are
    # also among its 10 nearest neighbours *on screen* fell from 0.234 to
    # 0.083 at the top20 level (0.085 -> 0.035 across all 6,537). Shrinking
    # the distance to real neighbours doesn't help when the distance to
    # everyone else shrinks just as much.
    xy = normalize_coords(compress_outliers(compute_layout(embeddings, seed=seed)))

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




def _plain_text(value: str, limit: int = 70) -> str:
    """Commons credits arrive as HTML -- strip it to something displayable.

    The Artist field is free-form wiki markup, so a credit can be an
    anchor wrapped in <bdi> with a title attribute. Rendering that as-is
    would be wrong and storing it costs several times what the name does.
    """
    if not value:
        return ""
    text = html.unescape(re.sub(r"<[^>]+>", " ", value))
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit]


def _detected_size(record: dict) -> tuple[float, float]:
    """The pixel size the face boxes are expressed in.

    Not the image's own width/height: the collector detects on a Commons
    thumbnail, and ?width=800 is served from a per-image bucket list, so the
    result is 960 or 727 or 509 rather than 800. Normalising the boxes by the
    original's dimensions put every face up and to the left of the real one,
    by the ratio between the two -- which is why a crop could land on the
    backdrop beside someone's head.

    Records collected before det_width was stored fall back to the original
    dimensions, which is what they were already being scaled by.
    """
    width = record.get("det_width") or record.get("width") or 1
    height = record.get("det_height") or record.get("height") or 1
    return float(width), float(height)


def _photo_ref(face) -> dict:
    """A face as the frontend needs it: which Commons file, and where in it.

    Stores the bare filename rather than a URL -- the frontend rebuilds a
    Special:FilePath link, which is both shorter here and lets the display
    size be chosen at render time. The box is normalised to fractions of
    the image so the frontend can crop to the face without knowing the
    pixel size of whatever thumbnail it ends up being served.
    """
    record, i = face.record, face.index
    box = record["faces"][i]["bbox"] if "faces" in record else [0, 0, 1, 1]
    width, height = _detected_size(record)
    return {
        "f": record.get("title", "").replace("File:", ""),
        "b": [round(box[0] / width, 4), round(box[1] / height, 4),
              round(box[2] / width, 4), round(box[3] / height, 4)],
        # The image's own aspect ratio. Needed to crop to the face and not
        # guessable from the box: detector boxes are markedly taller than
        # wide (around 1:1.85 in this data), so inferring the image's shape
        # from the face's put the crop on empty background.
        "a": round(width / height, 3) if height else 1.0,
        "c": _plain_text(record.get("creator") or ""),
        "l": _plain_text(record.get("license") or "", 40),
    }


# What a photo has to clear to be shown, measured over the 19,286 accepted
# faces in the full collection. Together these keep 97% of people supplied
# with at least one picture.
MAX_SOURCE_WIDTH = 2400     # the cap photos.ts uses when asking Commons
MIN_DISPLAY_PIXELS = 150    # below this a crop renders soft (p10 is 114)
MIN_DISPLAY_DET_SCORE = 0.75  # a face turned away or occluded scores lower
MIN_DISPLAY_LIKENESS = 0.60   # and so does one barely recognisable as them

# Above this, two "people" are one gallery under two names rather than two
# faces that happen to look alike. The strongest genuine pair measured is
# 0.60 (Marie Antoinette and Mary, Queen of Scots, both from painted
# portraits); real duplicates sit at 0.78 and above.
DUPLICATE_IDENTITY = 0.70


def build_dataset_from_commons(
    directory: Path, *, k: int, seed: int, out_dir: Path, similar_k: int = 15,
    photos_per_person: int = 6,
) -> dict:
    """Build the site's dataset from the Wikimedia Commons collection.

    One graph, not a stack of popularity levels: the fame slider filtered on
    how many photos a person had, which was only ever a proxy, and the
    Commons collector gives everyone a comparable handful.

    Alongside the graph this writes photos.json -- the real face crops, and
    which pair of photographs to show when two people are compared. It is a
    separate file because the comparison view is opened rarely and it would
    otherwise be dead weight in every first page load.
    """
    from .commons_embeddings import (
        best_matching_faces, load_commons_people, person_prototype,
    )

    people = load_commons_people(directory)
    prototypes, accepted = {}, {}
    for name, images in sorted(people.items()):
        result = person_prototype(name, images)
        if result is not None:
            prototypes[name], accepted[name] = result
    names = list(prototypes)
    if len(names) < 2:
        raise ValueError(f"only {len(names)} usable people in {directory}")

    # Subtract the average face before comparing anyone.
    #
    # ArcFace vectors all share a large "generic human face" component, so raw
    # cosines sit in a narrow band and the ranking is dominated by whatever is
    # common to everyone rather than by what makes two people look alike.
    # Centring on the mean of the population removes it, which is what the
    # method-evaluation notebook scored its candidates in.
    mean_face = np.stack([prototypes[n] for n in names]).mean(axis=0)

    def centred(vectors: np.ndarray) -> np.ndarray:
        out = vectors - mean_face
        return out / np.maximum(np.linalg.norm(out, axis=-1, keepdims=True), 1e-9)

    embeddings = centred(np.stack([prototypes[n] for n in names]))

    # Drop identities that are really the same gallery twice.
    #
    # Commons hands the same person back under two spellings ("Catherine
    # O'Hara" and "Catherine O_Hara" are byte-for-byte the same 13 photos),
    # and a search occasionally returns somebody else entirely -- every one
    # of "Debbie Rowe"'s photos is Elon Musk. Both surface as a pair scoring
    # far above anything two different faces reach, so one threshold catches
    # them. The loser is the thinner gallery; the threshold is set well above
    # the strongest genuine pair so that lookalikes and relatives survive.
    duplicate = np.triu(embeddings @ embeddings.T, 1) > DUPLICATE_IDENTITY
    drop = set()
    for a, b in zip(*np.nonzero(duplicate)):
        loser = a if len(accepted[names[a]]) < len(accepted[names[b]]) else b
        drop.add(int(loser))
    if drop:
        keep = [i for i in range(len(names)) if i not in drop]
        print(f"dropping {len(drop)} duplicate identities: "
              + ", ".join(sorted(names[i] for i in drop)))
        names = [names[i] for i in keep]
        embeddings = embeddings[keep]

    neighbor_idx, sim = build_knn(embeddings, k=max(k, similar_k))
    edges = mutual_knn_edges(neighbor_idx[:, :k], sim[:, :k])
    similar = directed_similar_lists(neighbor_idx[:, :similar_k], sim[:, :similar_k])
    deg = mutual_degrees(edges, n=len(names))
    xy = normalize_coords(compress_outliers(compute_layout(embeddings, seed=seed)))

    # The faces shown for each person, best first.
    #
    # Ranked by how many pixels of face the browser can actually fetch, which
    # is what decides whether a crop looks sharp. The obvious measure -- how
    # much of the frame the face fills -- is not the same thing and gets it
    # backwards often enough to matter: a head filling half of a 400px image
    # is 200px of face and renders soft, while a tenth of a 4000px image is
    # twice that and renders crisp.
    #
    # Then the face has to be worth showing at all. A low detector score is a
    # face turned away, half out of frame or behind something, and a face
    # unlike the person's own prototype is the same story from the other
    # side. Both are useless in a side-by-side comparison, where the whole
    # point is to see two faces.
    #
    # Anyone left with nothing keeps their best photo anyway: a missing
    # picture is worse than a soft one, and 27 of 965 people have no image
    # clearing the bar.
    def display_pixels(face):
        """Width of the face, in pixels of the largest source we can request."""
        box = face.record["faces"][face.index]["bbox"] if "faces" in face.record else None
        if not box:
            return 0.0
        width, _ = _detected_size(face.record)
        source = min(MAX_SOURCE_WIDTH, face.record.get("width") or width)
        return ((box[2] - box[0]) / width) * source

    def worth_showing(face, prototype):
        detail = face.record["faces"][face.index] if "faces" in face.record else {}
        return (display_pixels(face) >= MIN_DISPLAY_PIXELS
                and detail.get("det_score", 0.0) >= MIN_DISPLAY_DET_SCORE
                and float(face.embedding @ prototype) >= MIN_DISPLAY_LIKENESS)

    def ranked(name):
        usable = [f for f in accepted[name] if worth_showing(f, prototypes[name])]
        pool = usable or accepted[name]
        return sorted(pool, key=lambda f: -display_pixels(f))[:photos_per_person]

    photos = {str(i): [_photo_ref(f) for f in ranked(n)] for i, n in enumerate(names)}
    by_index = {n: ranked(n) for n in names}

    nodes = [{
        "id": i,
        "name": name,
        "x": round(float(xy[i, 0]), 1),
        "y": round(float(xy[i, 1]), 1),
        "deg": deg[i],
        "faces": len(accepted[name]),
    } for i, name in enumerate(names)]

    # For every pair the sidebar can offer, which two photographs actually
    # match -- worked out here so the browser never needs the embeddings.
    pairs: dict[str, list] = {}
    for i, name in enumerate(names):
        row = []
        for other, weight in similar[i]:
            match = best_matching_faces(by_index[name], by_index[names[other]], mean_face)
            if match is None:
                row.append([other, weight, 0, 0])
                continue
            _, fa, fb = match
            # Identity, not equality: a PersonFace holds a numpy array, so
            # list.index() compares arrays and raises on the ambiguous truth
            # value rather than finding the element.
            row.append([
                other, weight,
                next(x for x, f in enumerate(by_index[name]) if f is fa),
                next(x for x, f in enumerate(by_index[names[other]]) if f is fb),
            ])
        pairs[str(i)] = row

    graph = {
        "meta": {"version": f"commons-{directory.name}", "count": len(names), "k": k},
        "nodes": nodes,
        "edges": [[a, b, w] for a, b, w in edges],
        "similar": pairs,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "graph.json").write_text(json.dumps(graph), encoding="utf-8")
    (out_dir / "photos.json").write_text(json.dumps(photos), encoding="utf-8")
    return graph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--commons", type=Path,
        help="directory of per-person JSON from the Commons collector (current source)",
    )
    source.add_argument(
        "--embeddings", type=Path,
        help="path to a legacy IMDB-WIKI .npz (names + E arrays)",
    )
    parser.add_argument("--k", type=int, default=8, help="kNN neighbors per node")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    if args.commons:
        graph = build_dataset_from_commons(
            args.commons, k=args.k, seed=args.seed, out_dir=args.out
        )
        print(f"Wrote {len(graph['nodes'])} nodes, {len(graph['edges'])} edges "
              f"to {args.out}/graph.json (+ photos.json)")
        return

    graphs = build_dataset_from_embeddings(
        args.embeddings, k=args.k, seed=args.seed, out_dir=args.out
    )
    for label, g in graphs.items():
        print(f"[{label}] Wrote {len(g['nodes'])} nodes, {len(g['edges'])} edges "
              f"to {args.out}/graph-{label}.json")


if __name__ == "__main__":
    main()
