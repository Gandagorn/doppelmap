"""Backfill the thumbnail size each face box was measured in.

The collector detects faces on a Commons thumbnail but stored only the
*original* image's width and height, so build_dataset normalised the boxes
by the wrong number and every face in an image larger than the thumbnail
landed up and to the left of the real one.

The scale cannot be worked out from the stored record -- ?width=800 is
served from a per-image bucket list, giving 960 or 727 or 509 -- but the
Commons API will report the thumbnail's dimensions for a given request
width, 50 titles at a time, so this needs no image downloads.

Newly collected files already carry det_width/det_height and are skipped.

    python repair_face_boxes.py ../../data/faces
"""
import json
import sys
import time
from pathlib import Path

import requests

API = "https://commons.wikimedia.org/w/api.php"
THUMB_WIDTH = 800          # must match the collector's THUMB_WIDTH
BATCH = 50                 # the API's limit for a titles= query
USER_AGENT = "doppelmap/1.0 (https://github.com/Gandagorn/doppelmap)"


def thumb_sizes(session, titles):
    """{title: (width, height)} for each title's THUMB_WIDTH thumbnail."""
    response = session.get(API, timeout=60, params={
        "action": "query", "format": "json", "prop": "imageinfo",
        "iiprop": "url|size", "iiurlwidth": THUMB_WIDTH,
        "titles": "|".join(titles),
    })
    response.raise_for_status()
    sizes = {}
    for page in response.json().get("query", {}).get("pages", {}).values():
        info = (page.get("imageinfo") or [{}])[0]
        # A file smaller than THUMB_WIDTH is served as-is and reports no
        # thumb dimensions, so fall back to its own size.
        width = info.get("thumbwidth") or info.get("width")
        height = info.get("thumbheight") or info.get("height")
        if width and height:
            sizes[page["title"]] = (int(width), int(height))
    return sizes


def repair(directory: Path) -> None:
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    files = sorted(directory.glob("*.json"))
    fixed = people = 0

    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        images = payload["images"] if isinstance(payload, dict) else payload
        todo = [r for r in images
                if r.get("faces") and not r.get("det_width") and r.get("title")]
        if not todo:
            continue

        sizes = {}
        for start in range(0, len(todo), BATCH):
            titles = [r["title"] for r in todo[start:start + BATCH]]
            for attempt in range(5):
                try:
                    sizes.update(thumb_sizes(session, titles))
                    break
                except requests.RequestException:
                    time.sleep(2 * 2 ** attempt)
            time.sleep(0.2)

        for record in todo:
            size = sizes.get(record["title"])
            if not size:
                continue
            width, height = size
            # The collector's last retry falls back to the full-size
            # original, so a box reaching past the thumbnail was measured on
            # the original instead. Trust the boxes, not the assumption.
            widest = max(f["bbox"][2] for f in record["faces"])
            tallest = max(f["bbox"][3] for f in record["faces"])
            if widest > width + 1 or tallest > height + 1:
                width = record.get("width") or width
                height = record.get("height") or height
            record["det_width"], record["det_height"] = int(width), int(height)
            fixed += 1

        temp = path.with_suffix(".json.tmp")
        temp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        temp.replace(path)
        people += 1
        print(f"  {path.stem}: {len(todo)} images", flush=True)

    print(f"\nrepaired {fixed} images across {people} people")


if __name__ == "__main__":
    repair(Path(sys.argv[1] if len(sys.argv) > 1 else "../../data/faces"))
