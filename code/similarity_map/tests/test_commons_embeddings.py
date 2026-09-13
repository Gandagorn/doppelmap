import json

import numpy as np

from similarity_map.pipeline.commons_embeddings import (
    _identifying_token,
    load_commons_embeddings,
    person_prototype,
    records_depicting,
)


def _record(title, emb):
    return {"title": title, "emb": list(map(float, emb))}


def _unit(rng, n=1):
    v = rng.normal(size=(n, 512)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def _cluster(rng, base, n, jitter=0.02):
    """n near-identical vectors around `base` -- photos of one person."""
    v = base + rng.normal(size=(n, 512)).astype(np.float32) * jitter
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_identifying_token_prefers_the_most_distinctive_word():
    # Surname, not first name: "Tom" is shared across the dataset, so a
    # Tom Holland photo must not satisfy a Tom Cruise query.
    assert _identifying_token("Tom Cruise") == "cruise"
    # Regnal names have no surname -- last-token would give the numeral.
    assert _identifying_token("Charles III") == "charles"
    assert _identifying_token("Elizabeth II") == "elizabeth"
    assert _identifying_token("Zendaya") == "zendaya"
    assert _identifying_token("John F. Kennedy") == "kennedy"


def test_records_depicting_drops_other_people_from_the_same_photoset():
    # The real failure: a Commons search for "Colin Hanks" returned five
    # Michael Cera photos from one Flickr set and only two of Colin Hanks.
    records = [
        _record("File:Colin Hanks in 2015.jpg", [0] * 512),
        _record("File:Colin Hanks SXSW 2015.jpg", [0] * 512),
        _record("File:Michael Cera (6986349341).jpg", [0] * 512),
        _record("File:Kieran Culkin (6840233606).jpg", [0] * 512),
    ]
    kept = records_depicting("Colin Hanks", records)
    assert [r["title"] for r in kept] == [
        "File:Colin Hanks in 2015.jpg",
        "File:Colin Hanks SXSW 2015.jpg",
    ]


def test_person_prototype_averages_agreeing_photos():
    rng = np.random.default_rng(0)
    base = _unit(rng)[0]
    records = [_record(f"p{i}", v) for i, v in enumerate(_cluster(rng, base, 6))]

    result = person_prototype(records)

    assert result is not None
    prototype, kept = result
    assert len(kept) == 6
    assert np.isclose(np.linalg.norm(prototype), 1.0, atol=1e-5)
    assert float(prototype @ base) > 0.9


def test_person_prototype_ignores_a_minority_of_odd_photos():
    # Two unrelated faces mixed into a gallery of eight must not move the
    # prototype away from the real person.
    rng = np.random.default_rng(1)
    base = _unit(rng)[0]
    good = _cluster(rng, base, 8)
    intruders = _unit(rng, 2)
    records = [_record(f"p{i}", v) for i, v in enumerate(np.vstack([good, intruders]))]

    prototype, kept = person_prototype(records)

    assert len(kept) == 8
    assert float(prototype @ base) > 0.9


def test_person_prototype_rejects_a_contaminated_gallery():
    # No majority agrees with anything -- ten unrelated faces. Returning a
    # confident prototype here is the dangerous case: it would ship as
    # that person with the wrong face.
    rng = np.random.default_rng(2)
    records = [_record(f"p{i}", v) for i, v in enumerate(_unit(rng, 10))]
    assert person_prototype(records) is None


def test_person_prototype_rejects_a_thin_gallery():
    rng = np.random.default_rng(3)
    records = [_record(f"p{i}", v) for i, v in enumerate(_unit(rng, 2))]
    assert person_prototype(records) is None


def test_load_commons_embeddings_end_to_end(tmp_path):
    rng = np.random.default_rng(4)
    alice, bob = _unit(rng, 2)

    (tmp_path / "Alice Smith.json").write_text(
        json.dumps([
            _record(f"File:Alice Smith {i}.jpg", v)
            for i, v in enumerate(_cluster(rng, alice, 5))
        ]),
        encoding="utf-8",
    )
    # Bob's gallery is mostly someone else's photoset; only two real Bob
    # photos survive the title check, which is under MIN_IMAGES.
    (tmp_path / "Bob Jones.json").write_text(
        json.dumps(
            [_record(f"File:Bob Jones {i}.jpg", v) for i, v in enumerate(_cluster(rng, bob, 2))]
            + [_record(f"File:Someone Else {i}.jpg", v) for i, v in enumerate(_unit(rng, 6))]
        ),
        encoding="utf-8",
    )
    (tmp_path / "Empty Person.json").write_text("[]", encoding="utf-8")

    prototypes, counts = load_commons_embeddings(tmp_path)

    assert set(prototypes) == {"Alice Smith"}
    assert counts == {"Alice Smith": 5}
    assert np.isclose(np.linalg.norm(prototypes["Alice Smith"]), 1.0, atol=1e-5)
