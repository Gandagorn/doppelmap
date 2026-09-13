import json

import numpy as np

from similarity_map.pipeline.commons_embeddings import (
    _identifying_token,
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

    prototype, used = person_prototype("Colin Hanks", images)

    assert used == 2
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

    prototype, used = person_prototype("Tom Hanks", solo + group)

    assert used == 5  # four solo shots plus Tom's face from the group photo
    assert float(prototype @ tom) > 0.9
    assert float(prototype @ costar) < 0.5


def test_prototype_keeps_a_person_whose_images_never_name_them():
    # No title match at all -- the gallery is still used rather than the
    # person being dropped. Filtering people out is a last resort.
    rng = np.random.default_rng(2)
    base = _unit(rng)[0]
    images = [_img(f"File:IMG_{i}.jpg", v) for i, v in enumerate(_cluster(rng, base, 4))]

    prototype, used = person_prototype("Someone Unnamed", images)

    assert used == 4
    assert float(prototype @ base) > 0.9


def test_prototype_keeps_a_person_whose_gallery_does_not_agree():
    # Two named photos that don't look like each other still produce a
    # prototype rather than dropping the person.
    rng = np.random.default_rng(3)
    images = [_img(f"File:Jane Doe {i}.jpg", v) for i, v in enumerate(_unit(rng, 2))]

    result = person_prototype("Jane Doe", images)

    assert result is not None
    prototype, used = result
    assert used >= 1
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

    prototype, used = person_prototype("Old Schema", images)

    assert used == 3
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
