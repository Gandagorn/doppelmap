"""Loads per-image face embeddings collected from Wikimedia Commons.

The collector writes one JSON file per person, named after them, holding
image records with 512-d L2-normalized face embeddings plus provenance
(Commons URL, licence, creator). That is a different shape from the older
IMDB-WIKI pipeline, which handed over a .npz already reduced to one
prototype per person, so the reduction happens here instead.

Everything downstream is unchanged: this emits the same
(embeddings_by_name, count_by_name) pair that real_embeddings.py does,
which is the seam build_dataset.py consumes.
"""
import json
from pathlib import Path
from typing import NamedTuple

import numpy as np

# Cosine similarity above which two faces are taken to be the same person.
# 0.35 is ArcFace's conventional same-identity floor.
SAME_PERSON = 0.35
# Below this many usable faces there is nothing to average; this is the
# only reason a person is dropped.
MIN_FACES = 2


def _identifying_token(name: str) -> str:
    """The most distinctive word in a name, lowercased.

    The longest token beats "the last one" for regnal names (Charles III ->
    "charles", not "iii") and beats "any token" for shared first names
    (Tom Cruise -> "cruise", so a Tom Holland photo can't match).
    """
    tokens = [t.strip(".,'\"") for t in name.split()]
    return max(tokens, key=len).lower() if tokens else name.lower()


def names_the_person(name: str, record: dict) -> bool:
    """Does this image's title or URL name the person?

    Commons filenames are descriptive and human-curated, so a title or URL
    carrying the person's name is strong evidence the person is in the
    frame -- far stronger than anything recoverable from the pixels alone
    when a gallery is mixed.
    """
    token = _identifying_token(name)
    haystack = f"{record.get('title', '')} {record.get('original_url', '')}"
    return token in haystack.replace("_", " ").lower()


class PersonFace(NamedTuple):
    """One face that has been accepted as belonging to a person."""
    embedding: np.ndarray
    record: dict          # the image it came from, with title/url/licence
    index: int            # which face within that image


def _faces_of(record: dict) -> list[list[float]]:
    """Every face embedding in one image record.

    Handles both collector schemas: the newer one stores `faces: [{bbox,
    emb}, ...]` so a group photo contributes each face separately, while
    the older one stored a single `emb` per image.
    """
    faces = record.get("faces")
    if isinstance(faces, list):
        return [f["emb"] for f in faces if f.get("emb")]
    return [record["emb"]] if record.get("emb") else []


def load_commons_people(directory: Path) -> dict[str, list[dict]]:
    """Reads every *.json in `directory`; the filename is the person.

    Accepts either a bare list of image records or a {"name", "images"}
    object, which is what the newer collector writes.
    """
    people = {}
    for path in sorted(Path(directory).glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        records = payload.get("images", []) if isinstance(payload, dict) else payload
        if records:
            people[path.stem] = records
    return people


def person_prototype(
    name: str, images: list[dict]
) -> tuple[np.ndarray, list[PersonFace]] | None:
    """Reduces one person's images to a single unit vector.

    Images whose title or URL names the person are the anchor: they decide
    *which* face is this person, which matters because a gallery is rarely
    clean. A Commons search for "Colin Hanks" returned five Michael Cera
    photos from one Flickr set against two real ones, and picking the
    face that recurs most across the whole gallery confidently produced
    Cera. Anchoring on the named images produces Colin.

    Within the anchors the medoid still does real work, because a
    correctly-named photo is often a group shot ("Tom Hanks and Steven
    Spielberg") -- the face that recurs across the anchors is the subject,
    and the co-stars drop out.

    Returns the prototype and the faces that were accepted as this
    person's. Those survivors matter downstream, not just the count: a
    group photo naming two of our people contributes every one of its
    faces to both of them, so anyone comparing photo-to-photo has to work
    from the filtered set. Comparing the raw anchored faces of Charles III
    and Elizabeth II found the same crop of the same family photograph on
    both sides and scored it 1.000 -- a face against itself.

    Nothing is discarded for being an odd gallery: a person is only
    returned as None when there are too few faces to average at all.
    """
    anchored = [im for im in images if names_the_person(name, im)]
    # Falling back to the full gallery rather than giving up: a thin or
    # oddly-titled set still beats losing the person entirely.
    pool = anchored or images

    candidates = [
        (np.asarray(emb, dtype=np.float32), record, i)
        for record in pool
        for i, emb in enumerate(_faces_of(record))
    ]
    if len(candidates) < MIN_FACES:
        return None

    embeddings = np.stack([c[0] for c in candidates])
    similarity = embeddings @ embeddings.T
    medoid = int(similarity.sum(axis=1).argmax())
    keep = similarity[medoid] > SAME_PERSON
    if not keep.any():
        keep[medoid] = True

    prototype = embeddings[keep].mean(axis=0)
    prototype /= np.linalg.norm(prototype)
    accepted = [PersonFace(*candidates[i]) for i in np.flatnonzero(keep)]
    return prototype, accepted


def load_commons_embeddings(
    directory: Path,
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """Directory of per-person JSON -> (prototype by name, face count by name).

    Same return shape as real_embeddings.load_real_embeddings plus
    load_popularity, so build_dataset can take either source.
    """
    prototypes: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {}
    for name, images in load_commons_people(directory).items():
        result = person_prototype(name, images)
        if result is None:
            continue
        prototype, accepted = result
        prototypes[name] = prototype
        counts[name] = len(accepted)
    return prototypes, counts


def best_matching_faces(
    a: list[PersonFace], b: list[PersonFace]
) -> tuple[float, PersonFace, PersonFace] | None:
    """The most similar photo-to-photo pairing between two people.

    What the comparison view shows: not two arbitrary portraits, but the
    two actual photographs the model considers closest.

    Pass the *accepted* faces from person_prototype, never the raw gallery,
    and note that faces from the same source image are skipped. Both guards
    exist because a photo captioned with two of our people is attributed to
    both: without them Charles III and Elizabeth II match at 1.000 on one
    shared family portrait, and Barron and Donald Trump at 0.766 on a face
    of Donald sitting inside Barron's gallery.
    """
    if not a or not b:
        return None
    similarity = np.stack([f.embedding for f in a]) @ np.stack([f.embedding for f in b]).T
    for i, fa in enumerate(a):
        for j, fb in enumerate(b):
            if fa.record.get("title") and fa.record["title"] == fb.record.get("title"):
                # -inf, not -1: cosine similarity is legitimately negative
                # sometimes, and a real pair must never be mistaken for a
                # masked one.
                similarity[i, j] = -np.inf
    i, j = np.unravel_index(int(similarity.argmax()), similarity.shape)
    if not np.isfinite(similarity[i, j]):
        return None                    # every pairing came from a shared image
    return float(similarity[i, j]), a[i], b[j]
