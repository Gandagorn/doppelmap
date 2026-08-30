"""Placeholder avatar thumbnails (colored initials) — stand in for real
face crops until the pipeline has actual images to embed.
"""
import hashlib
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

THUMB_SIZE = 96
_PALETTE = [
    (230, 126, 34), (41, 128, 185), (39, 174, 96), (142, 68, 173),
    (192, 57, 43), (22, 160, 133), (211, 84, 0), (44, 62, 80),
]


def _initials(name: str) -> str:
    parts = [p for p in name.split() if p]
    if not parts:
        return "?"
    if len(parts) == 1:
        return parts[0][:2].upper()
    return (parts[0][0] + parts[-1][0]).upper()


def _color_for(name: str) -> tuple:
    digest = hashlib.sha256(name.encode("utf-8")).digest()
    return _PALETTE[digest[0] % len(_PALETTE)]


def generate_thumbnail(name: str, out_path: Path) -> None:
    """Writes a THUMB_SIZE x THUMB_SIZE WebP avatar with the person's
    initials on a deterministic (name-hashed) background color.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (THUMB_SIZE, THUMB_SIZE), _color_for(name))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default(size=44)
    except TypeError:
        # Pillow < 10.1 doesn't support the size kwarg.
        font = ImageFont.load_default()
    text = _initials(name)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(
        ((THUMB_SIZE - text_w) / 2 - bbox[0], (THUMB_SIZE - text_h) / 2 - bbox[1]),
        text,
        fill=(255, 255, 255),
        font=font,
    )
    img.save(out_path, "WEBP", quality=80)
