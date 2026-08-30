from PIL import Image
from similarity_map.pipeline.thumbnails import (
    generate_thumbnail,
    THUMB_SIZE,
    _initials,
    _color_for,
)


def test_generate_thumbnail_creates_expected_file(tmp_path):
    out_path = tmp_path / "t" / "0.webp"
    generate_thumbnail("Tom Hanks", out_path)
    assert out_path.exists()
    img = Image.open(out_path)
    assert img.size == (THUMB_SIZE, THUMB_SIZE)


def test_initials_two_word_name():
    assert _initials("Tom Hanks") == "TH"


def test_initials_single_word_name():
    assert _initials("Beyonce") == "BE"


def test_color_is_deterministic():
    assert _color_for("Tom Hanks") == _color_for("Tom Hanks")
