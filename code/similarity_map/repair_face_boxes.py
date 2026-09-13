"""Backfill the thumbnail size each face box was measured in.

The collector detects faces on a Commons thumbnail but older versions stored
only the *original* image's width and height, so build_dataset normalised the
boxes by the wrong number: every face in an image bigger than the thumbnail
landed up and to the left of the real one, and shrunk by the same ratio.

Commons answers CELL 5's ?width=800 request from a per-image bucket list, and
the rule is exact: it serves min(960, original width), scaling the height to
match, and anything narrower than that is served untouched. Verified against
59 images spanning 349px to 4608px originals -- 59 correct, 0 wrong. So the
size can be recomputed offline and this makes no network requests at all.

An earlier version of this script asked the API instead, via iiurlwidth. That
was wrong for every image tested: the API happily reports an *upscaled* 800px
thumbnail for a 444px original, and reports 800 where Commons really serves
960. Do not reach for the API here.

Records written by a current collector already carry det_width measured
straight off the decoded image; those are authoritative and left alone.

    python repair_face_boxes.py ../../data/faces
"""
import json
import sys
from pathlib import Path

THUMB_BUCKET = 960         # what Commons serves for CELL 5's ?width=800


def detected_size(record: dict) -> tuple[int, int] | None:
    """The pixel size this record's face boxes were measured in."""
    width, height = record.get("width"), record.get("height")
    if not width or not height:
        return None
    served_width = min(THUMB_BUCKET, width)
    served_height = round(height * served_width / width)

    # The collector's last retry falls back to the full-size original, so a
    # box reaching past the thumbnail was measured on the original instead.
    # Trust the boxes over the rule.
    boxes = record.get("faces") or []
    if any(f["bbox"][2] > served_width + 1 or f["bbox"][3] > served_height + 1
           for f in boxes):
        return int(width), int(height)
    return int(served_width), int(served_height)


def repair(directory: Path) -> None:
    fixed = people = 0
    for path in sorted(directory.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        images = payload["images"] if isinstance(payload, dict) else payload
        changed = False
        for record in images:
            if not record.get("faces") or record.get("det_width"):
                continue
            size = detected_size(record)
            if not size:
                continue
            record["det_width"], record["det_height"] = size
            fixed += 1
            changed = True
        if not changed:
            continue
        temp = path.with_suffix(".json.tmp")       # as flush() does, so an
        temp.write_text(json.dumps(payload, ensure_ascii=False),
                        encoding="utf-8")          # interrupted run cannot
        temp.replace(path)                         # leave a half-written file
        people += 1
    print(f"repaired {fixed} images across {people} people")


if __name__ == "__main__":
    repair(Path(sys.argv[1] if len(sys.argv) > 1 else "../../data/faces"))
