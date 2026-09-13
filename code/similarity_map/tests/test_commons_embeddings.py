import json

import numpy as np
import pytest

from similarity_map.pipeline.commons_embeddings import (
    PersonFace,
    _identifying_token,
    best_matching_faces,
    load_commons_embeddings,
    names_the_person,
    person_prototype,
)


def _unit(rng, n=1):
    v = rng.normal(size=(n, 512)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def _cluster(rng, base, n, jitter=0.02):
    """n near-identical vectors around `base` -- photos of one person."""
    v = base + rng.normal(size=(n, 512)).astype(np.float32) * jitter
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def _img(title, *embs, url=""):
    return {"title": title, "original_url": url, "faces": [{"emb": list(map(float, e))} for e in embs]}


def test_identifying_token_prefers_the_most_distinctive_word():
    # Surname, not first name: "Tom" is shared across the dataset, so a Tom
    # Holland photo must not satisfy a Tom Cruise query.
    assert _identifying_token("Tom Cruise") == "cruise"
    # Regnal names have no surname -- last-token would give the numeral.
    assert _identifying_token("Charles III") == "charles"
    assert _identifying_token("Zendaya") == "zendaya"
    assert _identifying_token("John F. Kennedy") == "kennedy"


def test_names_the_person_matches_title_or_url():
    rec = {"title": "File:Colin Hanks SXSW 2015.jpg", "original_url": ""}
    assert names_the_person("Colin Hanks", rec)
    # Commons URLs use underscores, which must not defeat the match.
    url_only = {"title": "", "original_url": "https://upload.wikimedia.org/a/Colin_Hanks_in_2015.jpg"}
    assert names_the_person("Colin Hanks", url_only)
    other = {"title": "File:Michael Cera (6986349341).jpg", "original_url": ""}
    assert not names_the_person("Colin Hanks", other)


def test_prototype_anchors_on_the_named_images_not_the_majority_face():
    # The real failure: a "Colin Hanks" gallery held five Michael Cera
    # photos from one Flickr set and two genuine ones. Taking the face that
    # recurs most across the whole gallery yields Cera; anchoring on the
    # images that actually name Colin yields Colin.
    rng = np.random.default_rng(0)
    colin, cera = _unit(rng, 2)
    images = [_img(f"File:Colin Hanks {i}.jpg", v) for i, v in enumerate(_cluster(rng, colin, 2))]
    images += [_img(f"File:Michael Cera {i}.jpg", v) for i, v in enumerate(_cluster(rng, cera, 5))]

    prototype, accepted = person_prototype("Colin Hanks", images)

    assert len(accepted) == 2
    assert float(prototype @ colin) > 0.9
    assert float(prototype @ cera) < 0.5


def test_prototype_drops_co_stars_from_a_correctly_named_group_photo():
    # "Tom Hanks and Steven Spielberg.jpg" legitimately names Tom, but
    # contributes two faces. The one recurring across his other named
    # photos is his; the co-star must not pull the prototype.
    rng = np.random.default_rng(1)
    tom, costar = _unit(rng, 2)
    solo = [_img(f"File:Tom Hanks {i}.jpg", v) for i, v in enumerate(_cluster(rng, tom, 4))]
    group = [_img("File:Tom Hanks and a co-star.jpg", _cluster(rng, tom, 1)[0], costar)]

    prototype, accepted = person_prototype("Tom Hanks", solo + group)

    assert len(accepted) == 5  # four solo shots plus Tom's face from the group
    assert float(prototype @ tom) > 0.9
    assert float(prototype @ costar) < 0.5


def test_prototype_keeps_a_person_whose_images_never_name_them():
    # No title match at all -- the gallery is still used rather than the
    # person being dropped. Filtering people out is a last resort.
    rng = np.random.default_rng(2)
    base = _unit(rng)[0]
    images = [_img(f"File:IMG_{i}.jpg", v) for i, v in enumerate(_cluster(rng, base, 4))]

    prototype, accepted = person_prototype("Someone Unnamed", images)

    assert len(accepted) == 4
    assert float(prototype @ base) > 0.9


def test_prototype_keeps_a_person_whose_gallery_does_not_agree():
    # Two named photos that don't look like each other still produce a
    # prototype rather than dropping the person.
    rng = np.random.default_rng(3)
    images = [_img(f"File:Jane Doe {i}.jpg", v) for i, v in enumerate(_unit(rng, 2))]

    result = person_prototype("Jane Doe", images)

    assert result is not None
    prototype, accepted = result
    assert len(accepted) >= 1
    assert np.isclose(np.linalg.norm(prototype), 1.0, atol=1e-5)


def test_prototype_returns_none_only_when_there_is_nothing_to_average():
    rng = np.random.default_rng(4)
    assert person_prototype("Solo Person", [_img("File:Solo Person.jpg", _unit(rng)[0])]) is None
    assert person_prototype("Nobody", []) is None


def test_reads_the_older_single_embedding_schema():
    # Earlier collector runs stored one `emb` per image instead of `faces`.
    rng = np.random.default_rng(5)
    base = _unit(rng)[0]
    images = [
        {"title": f"File:Old Schema {i}.jpg", "emb": list(map(float, v))}
        for i, v in enumerate(_cluster(rng, base, 3))
    ]

    prototype, accepted = person_prototype("Old Schema", images)

    assert len(accepted) == 3
    assert float(prototype @ base) > 0.9


def test_load_commons_embeddings_end_to_end(tmp_path):
    rng = np.random.default_rng(6)
    alice, bob = _unit(rng, 2)

    (tmp_path / "Alice Smith.json").write_text(
        json.dumps({
            "name": "Alice Smith",
            "images": [_img(f"File:Alice Smith {i}.jpg", v)
                       for i, v in enumerate(_cluster(rng, alice, 5))],
        }),
        encoding="utf-8",
    )
    # Bare-list payload, the older collector's shape.
    (tmp_path / "Bob Jones.json").write_text(
        json.dumps([_img(f"File:Bob Jones {i}.jpg", v)
                    for i, v in enumerate(_cluster(rng, bob, 3))]),
        encoding="utf-8",
    )
    (tmp_path / "Empty Person.json").write_text("[]", encoding="utf-8")

    prototypes, counts = load_commons_embeddings(tmp_path)

    assert set(prototypes) == {"Alice Smith", "Bob Jones"}
    assert counts == {"Alice Smith": 5, "Bob Jones": 3}
    assert np.isclose(np.linalg.norm(prototypes["Alice Smith"]), 1.0, atol=1e-5)


def test_accepted_faces_carry_their_source_image():
    # The comparison view needs the image behind each face -- its title,
    # URL and licence -- not just the vector.
    rng = np.random.default_rng(10)
    base = _unit(rng)[0]
    images = [_img(f"File:Jane Roe {i}.jpg", v, url=f"http://x/{i}.jpg")
              for i, v in enumerate(_cluster(rng, base, 3))]

    _, accepted = person_prototype("Jane Roe", images)

    assert {f.record["title"] for f in accepted} == {f"File:Jane Roe {i}.jpg" for i in range(3)}
    assert all(f.index == 0 for f in accepted)


def test_best_matching_faces_picks_the_closest_photo_pair():
    rng = np.random.default_rng(11)
    # Jitter stays small: at 512 dimensions even 0.1 of noise per component
    # dwarfs the shared direction, and the two people stop resembling each
    # other at all.
    look = _unit(rng)[0]                       # the resemblance they share
    a_faces = _cluster(rng, look, 3, jitter=0.05)
    b_faces = _cluster(rng, look, 3, jitter=0.05)
    a = [_img(f"File:A {i}.jpg", v) for i, v in enumerate(a_faces)]
    b = [_img(f"File:B {i}.jpg", v) for i, v in enumerate(b_faces)]
    _, fa = person_prototype("A", a)
    _, fb = person_prototype("B", b)

    score, best_a, best_b = best_matching_faces(fa, fb)

    every = [(float(x.embedding @ y.embedding), x, y) for x in fa for y in fb]
    assert score == pytest.approx(max(s for s, _, _ in every))
    assert best_a.record["title"].startswith("File:A")
    assert best_b.record["title"].startswith("File:B")


def test_best_matching_faces_ignores_a_photo_the_two_people_share():
    # The real failure: "Elizabeth, Philip, Charles and Anne.jpg" names two
    # of our people, so the same crop lands in both galleries and matches
    # itself at 1.000. A shared image can never be evidence of resemblance.
    rng = np.random.default_rng(12)
    shared_face, solo_a, solo_b = _unit(rng, 3)
    group = _img("File:Two Royals Together.jpg", shared_face)
    a = [group, _img("File:A solo 1.jpg", solo_a), _img("File:A solo 2.jpg", solo_a)]
    b = [group, _img("File:B solo 1.jpg", solo_b), _img("File:B solo 2.jpg", solo_b)]

    fa = [PersonFace(np.asarray(f["emb"], np.float32), im, 0) for im in a for f in im["faces"]]
    fb = [PersonFace(np.asarray(f["emb"], np.float32), im, 0) for im in b for f in im["faces"]]
    score, best_a, best_b = best_matching_faces(fa, fb)

    assert score < 0.99                        # not a face against itself
    assert best_a.record["title"] != best_b.record["title"]


def test_best_matching_faces_handles_empty_input():
    assert best_matching_faces([], []) is None


def test_best_matching_faces_returns_none_when_the_only_photo_is_shared():
    # Two people whose sole image is one group photo: every possible
    # pairing comes from that photo, so there is no honest comparison to
    # show and the view gets nothing rather than a face against itself.
    rng = np.random.default_rng(13)
    a_face, b_face = _unit(rng, 2)
    shared = _img("File:Both Of Them.jpg", a_face, b_face)
    fa = [PersonFace(np.asarray(shared["faces"][0]["emb"], np.float32), shared, 0)]
    fb = [PersonFace(np.asarray(shared["faces"][1]["emb"], np.float32), shared, 1)]

    assert best_matching_faces(fa, fb) is None


def test_best_matching_faces_allows_cross_pairing_between_two_group_photos():
    # Only pairings *within* one image are masked. If the same two people
    # appear in two different group photos, comparing one person in the
    # first against the other in the second is a real comparison of two
    # distinct photographs, and should be offered.
    rng = np.random.default_rng(15)
    a_face, b_face = _unit(rng, 2)
    one = _img("File:Together One.jpg", a_face, b_face)
    two = _img("File:Together Two.jpg", a_face, b_face)
    fa = [PersonFace(np.asarray(im["faces"][0]["emb"], np.float32), im, 0) for im in (one, two)]
    fb = [PersonFace(np.asarray(im["faces"][1]["emb"], np.float32), im, 1) for im in (one, two)]

    result = best_matching_faces(fa, fb)

    assert result is not None
    _, best_a, best_b = result
    assert best_a.record["title"] != best_b.record["title"]


def test_best_matching_faces_keeps_a_genuinely_negative_match():
    # A real pairing that happens to score below zero is still a real
    # pairing, and must not be confused with a masked one.
    rng = np.random.default_rng(14)
    v = _unit(rng)[0]
    fa = [PersonFace(v, _img("File:A.jpg", v), 0)]
    fb = [PersonFace(-v, _img("File:B.jpg", -v), 0)]

    score, _, _ = best_matching_faces(fa, fb)

    assert score == pytest.approx(-1.0, abs=1e-4)
