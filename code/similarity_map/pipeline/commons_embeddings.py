"""Loads per-image embeddings collected from Wikimedia Commons.

The Commons collector writes one JSON file per person, named after them,
holding a list of image records -- each with its own 512-d L2-normalized
embedding plus provenance (Commons URL, licence, creator). That is a
different shape from the older IMDB-WIKI pipeline, which handed over a
.npz already reduced to one prototype vector per person.

So the medoid/consensus reduction that used to happen in the notebook
happens here instead. Everything downstream is unchanged: this emits the
same (embeddings_by_name, count_by_name) pair that real_embeddings.py
does, which is the seam build_dataset.py consumes.
"""
import json
from pathlib import Path

import numpy as np

# Cosine similarity above which two photos are taken to be the same
# person. 0.35 is ArcFace's conventional same-identity floor.
SAME_PERSON = 0.35
# A gallery where fewer than this share of photos agree with the medoid is
# too contaminated to trust, so the person is dropped entirely.
MIN_CONSENSUS = 0.5
# Fewer than this many usable photos and the prototype is too thin.
MIN_IMAGES = 3


def _identifying_token(name: str) -> str:
    """The most distinctive word in a name, lowercased.

    Used to check an image actually depicts the person. The longest token
    beats "the last one" for regnal names (Charles III -> "charles", not
    "iii") and beats "any token" for first names shared across the
    dataset (Tom Cruise -> "cruise", so a Tom Holland photo can't match).
    """
    tokens = [t.strip(".,") for t in name.split()]
    return max(tokens, key=len).lower() if tokens else name.lower()


def load_commons_people(directory: Path) -> dict[str, list[dict]]:
    """Reads every *.json in `directory`; the filename is the person."""
    people = {}
    for path in sorted(Path(directory).glob("*.json")):
        records = json.loads(path.read_text(encoding="utf-8"))
        if records:
            people[path.stem] = records
    return people


def records_depicting(name: str, records: list[dict]) -> list[dict]:
    """Drops images whose Commons title doesn't mention the person.

    Commons search matches on photoset and event metadata, not just
    subjects, so a query returns other people photographed at the same
    event -- for "Colin Hanks" it returned five Michael Cera shots from
    one Flickr set against only two real Colin Hanks photos. The medoid
    then locks onto the majority face and confidently mislabels it. The
    collector filters on this now too, but caches predate that fix, so
    this stays here as a guard rather than an assumption.
    """
    token = _identifying_token(name)
    return [r for r in records if token in r.get("title", "").lower()]


def person_prototype(records: list[dict]) -> tuple[np.ndarray, list[dict]] | None:
    """Reduces one person's images to a single unit vector.

    Picks the medoid (the photo most similar to all the others, i.e. the
    most typical one), keeps everything that agrees with it, and averages
    those. Returns None when the gallery is too thin or too contaminated.
    Also returns the kept records, so callers can use their URLs and
    licences.
    """
    embeddings = np.array([r["emb"] for r in records if r.get("emb")], dtype=np.float32)
    usable = [r for r in records if r.get("emb")]
    if len(embeddings) < MIN_IMAGES:
        return None

    similarity = embeddings @ embeddings.T
    medoid = int(similarity.sum(axis=1).argmax())
    keep = similarity[medoid] > SAME_PERSON
    if keep.mean() < MIN_CONSENSUS:
        return None

    prototype = embeddings[keep].mean(axis=0)
    prototype /= np.linalg.norm(prototype)
    return prototype, [r for r, k in zip(usable, keep) if k]


def load_commons_embeddings(
    directory: Path,
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """Directory of per-person JSON -> (prototype by name, image count by name).

    Same return shape as real_embeddings.load_real_embeddings plus
    load_popularity, so build_dataset can use either source.
    """
    prototypes: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {}
    for name, records in load_commons_people(directory).items():
        result = person_prototype(records_depicting(name, records))
        if result is None:
            continue
        prototype, kept = result
        prototypes[name] = prototype
        counts[name] = len(kept)
    return prototypes, counts
