from similarity_map.pipeline.names import CELEBRITY_NAMES


def test_names_are_unique_and_nonempty():
    assert len(CELEBRITY_NAMES) >= 150
    assert len(CELEBRITY_NAMES) == len(set(CELEBRITY_NAMES))
    assert all(isinstance(n, str) and n.strip() for n in CELEBRITY_NAMES)
