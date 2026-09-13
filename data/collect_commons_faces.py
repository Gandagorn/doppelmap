# ============================================================================
# Doppelmap - Wikimedia Commons face collection
#
# Takes a list of people and produces one JSON per person holding every face
# found in their Commons images, each with a bounding box and a 512-d ArcFace
# embedding.
#
# Replaces the earlier collector, which downloaded and embedded in one
# sequential pass. Downloading is I/O-bound and embedding is GPU-bound, so
# running them in lockstep left the GPU idle during every download and the
# network idle during every inference. Here they are decoupled: a pool of
# download threads keeps a bounded queue full while the GPU thread drains it.
#
# Copy each cell into Colab in order. Set Runtime > Change runtime type > GPU
# first, or CELL 1 will refuse to continue.
# ============================================================================


# ==== CELL 1 - install and verify GPU ======================================
!pip install -q insightface onnxruntime-gpu==1.22.0 opencv-python-headless tqdm
!pip uninstall -y -q onnxruntime

import torch  # pins Colab's cuDNN into the process before ORT loads
import onnxruntime as ort

assert ort.get_device() == "GPU", "Runtime > Change runtime type > GPU"
assert "CUDAExecutionProvider" in ort.get_available_providers(), "no CUDA provider"
print("onnxruntime", ort.__version__, "on", torch.cuda.get_device_name(0))


# ==== CELL 2 - config ======================================================
import os

from google.colab import drive

drive.mount("/content/drive")

WORK = "/content/drive/MyDrive/Projects/doppelmap"
PEOPLE_FILE = f"{WORK}/people_top5000.json"   # <-- your pageview-ranked list
OUT_DIR = f"{WORK}/faces"                     # one JSON per person lands here
os.makedirs(OUT_DIR, exist_ok=True)

# Wikimedia blocks generic user agents. A descriptive one with contact info is
# required by their policy, and is the difference between this working and
# getting 403ed a few thousand requests in.
USER_AGENT = "doppelmap/1.0 (https://github.com/Gandagorn/doppelmap)"

MAX_IMAGES_PER_PERSON = 40
MAX_FACES_PER_IMAGE = 4       # group shots: keep the largest few, not all
MIN_DET_SCORE = 0.60          # detector confidence floor
MIN_FACE_PX = 60              # faces smaller than this embed poorly
DOWNLOAD_WORKERS = 16         # I/O bound, so well above core count
QUEUE_DEPTH = 64              # decoded images awaiting the GPU (RAM bound)
REQUEST_TIMEOUT = 20          # seconds per HTTP request
MAX_RETRIES = 3
MAX_IMAGE_BYTES = 25_000_000  # skip absurd originals

print("output ->", OUT_DIR)


# ==== CELL 3 - people list and name matching ===============================
import json
import unicodedata

with open(PEOPLE_FILE, encoding="utf-8") as fh:
    raw_people = json.load(fh)

# Accepts either ["Tom Hanks", ...] or [{"name": ..., "views": ...}, ...],
# so it fits whatever your pageview ranking step already emits.
if raw_people and isinstance(raw_people[0], dict):
    def _name_of(p):
        return p.get("name") or p.get("article", "").replace("_", " ")
    PEOPLE = [_name_of(p) for p in raw_people]
    VIEWS = {_name_of(p): p.get("views", 0) for p in raw_people}
else:
    PEOPLE = list(raw_people)
    VIEWS = {}

PEOPLE = [p for p in PEOPLE if p]
print(f"{len(PEOPLE)} people; first few: {PEOPLE[:5]}")


def strip_accents(text):
    return "".join(
        c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn"
    )


def name_tokens(name):
    """Lowercased, accent-free words worth matching on.

    Drops initials and particles ("J.", "van", "de"), which match almost
    anything, leaving only the distinctive parts of a name.
    """
    parts = [strip_accents(t).strip(".,'\"").lower() for t in name.split()]
    skip = {"van", "von", "der", "den", "del", "the", "jr", "sr"}
    return [t for t in parts if len(t) > 2 and t not in skip]


def title_mentions(name, title):
    """Cheap prefilter: could this Commons title plausibly show this person?

    Deliberately permissive. It only has to be cheaper than downloading the
    image -- the real identity decision happens afterwards from the face
    embeddings themselves, because titles mislead in both directions.
    Commons search returns other people photographed at the same event (a
    "Colin Hanks" query returned five Michael Cera shots from one Flickr
    set), while plenty of correctly-titled images are group photos.

    Matches on the longest token, usually the surname. Last-token breaks on
    regnal names (Charles III -> "iii"), and any-token would let a Tom
    Holland photo satisfy a Tom Cruise query.
    """
    tokens = name_tokens(name)
    if not tokens:
        return True
    key = max(tokens, key=len)
    return key in strip_accents(title).lower()


# ==== CELL 4 - discover candidate images ===================================
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import requests
from tqdm.auto import tqdm

_local = threading.local()


def session():
    """One requests.Session per thread.

    Sessions are not thread-safe, but one per thread keeps connections alive
    across that thread's many requests, which matters a lot at this volume.
    """
    s = getattr(_local, "s", None)
    if s is None:
        s = requests.Session()
        s.headers["User-Agent"] = USER_AGENT
        _local.s = s
    return s


def api_get(url, params):
    """GET with bounded retries and backoff.

    Degrades to None rather than raising: no single failure should be able
    to take down a run that spans hours.
    """
    for attempt in range(MAX_RETRIES):
        try:
            r = session().get(url, params=params, timeout=REQUEST_TIMEOUT)
            if r.status_code == 429:          # rate limited
                time.sleep(2 * (attempt + 1))
                continue
            if r.ok:
                return r.json()
        except requests.RequestException:
            pass
        time.sleep(1.5 * (attempt + 1))
    return None


def discover(name):
    """Commons images plausibly showing `name`, with their metadata.

    Search hits and imageinfo come back in one call, so this is a single
    round trip per person rather than one per image.
    """
    data = api_get(
        "https://commons.wikimedia.org/w/api.php",
        {
            "action": "query",
            "generator": "search",
            "gsrsearch": f"{name} filetype:bitmap",
            "gsrnamespace": "6",              # File: namespace
            "gsrlimit": str(MAX_IMAGES_PER_PERSON),
            "prop": "imageinfo",
            "iiprop": "url|size|mime|extmetadata",
            "format": "json",
        },
    )
    if not data:
        return []

    out = []
    for page in (data.get("query", {}).get("pages") or {}).values():
        info = (page.get("imageinfo") or [{}])[0]
        title = page.get("title", "")
        if not info.get("url") or not str(info.get("mime", "")).startswith("image/"):
            continue
        if (info.get("size") or 0) > MAX_IMAGE_BYTES:
            continue
        if not title_mentions(name, title):
            continue
        meta = info.get("extmetadata") or {}
        out.append({
            "title": title,
            "source_url": f"https://commons.wikimedia.org/wiki/{title.replace(' ', '_')}",
            "original_url": info["url"],
            "width": info.get("width"),
            "height": info.get("height"),
            "mime": info.get("mime"),
            "bytes": info.get("size"),
            "license": (meta.get("LicenseShortName") or {}).get("value"),
            "license_url": (meta.get("LicenseUrl") or {}).get("value"),
            "creator": (meta.get("Artist") or {}).get("value"),
        })
    return out


todo = [p for p in PEOPLE if not os.path.exists(f"{OUT_DIR}/{p}.json")]
print(f"{len(PEOPLE) - len(todo)} already collected, {len(todo)} to go")

candidates = {}
with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as pool:
    for person, found in zip(
        todo, tqdm(pool.map(discover, todo), total=len(todo), desc="discover")
    ):
        candidates[person] = found

print(f"{sum(len(v) for v in candidates.values())} candidate images "
      f"across {len(candidates)} people")


# ==== CELL 5 - download (threads) + embed (GPU) ============================
import queue

import cv2
import numpy as np
from insightface.app import FaceAnalysis

app = FaceAnalysis(
    name="buffalo_l",
    allowed_modules=["detection", "recognition"],
    providers=["CUDAExecutionProvider"],
)
app.prepare(ctx_id=0, det_size=(640, 640))


def fetch(job):
    """Download and decode one image. Runs on a download thread."""
    person, rec = job
    for attempt in range(MAX_RETRIES):
        try:
            r = session().get(rec["original_url"], timeout=REQUEST_TIMEOUT)
            if r.ok and r.content:
                img = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
                return person, rec, img
        except requests.RequestException:
            pass
        time.sleep(1.0 * (attempt + 1))
    return person, rec, None


def faces_in(img):
    """Every usable face in one image, largest first.

    All of them, not just the most central one. A correctly-titled photo is
    often a group shot, and choosing a face here would be guessing; keeping
    them all lets the later consensus step decide which face is actually
    this person, using the evidence of their other photos.
    """
    found = []
    for f in app.get(img):
        if f.det_score < MIN_DET_SCORE:
            continue
        x1, y1, x2, y2 = (float(v) for v in f.bbox)
        if min(x2 - x1, y2 - y1) < MIN_FACE_PX:
            continue
        found.append({
            "bbox": [round(v, 1) for v in (x1, y1, x2, y2)],
            "det_score": round(float(f.det_score), 3),
            # Rounded to 4 decimals: measured on real embeddings that shifts
            # cosine similarity by at most 1.7e-04, far below anything that
            # could reorder a ranking, while cutting each stored vector from
            # ~13.7 KB to ~3.5 KB. Across 5,000 people that is the difference
            # between roughly 2 GB and 500 MB on Drive.
            "emb": [round(v, 4) for v in f.normed_embedding.astype(float).tolist()],
        })
    found.sort(key=lambda d: -(d["bbox"][2] - d["bbox"][0]) * (d["bbox"][3] - d["bbox"][1]))
    return found[:MAX_FACES_PER_IMAGE]


jobs = [(p, rec) for p, recs in candidates.items() for rec in recs]
results = {p: [] for p in candidates}
pending = {p: len(recs) for p, recs in candidates.items()}
stats = {"ok": 0, "no_face": 0, "failed": 0, "faces": 0}


def flush(person):
    """Write one person's file as soon as their last image is processed.

    Per-person checkpointing is what makes a Colab disconnect survivable:
    CELL 4 skips anyone whose file already exists, so re-running picks up
    where this left off. The temp-then-rename keeps a killed runtime from
    leaving a half-written file that would then be skipped as "done".
    """
    path = f"{OUT_DIR}/{person}.json"
    payload = {
        "name": person,
        "views": VIEWS.get(person),
        "collected_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "images": results.get(person, []),
    }
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False)
    os.replace(tmp, path)
    results.pop(person, None)


work = queue.Queue(maxsize=QUEUE_DEPTH)


def producer():
    with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as pool:
        for item in pool.map(fetch, jobs):
            work.put(item)        # blocks when full, throttling to GPU speed
    work.put(None)


threading.Thread(target=producer, daemon=True).start()

with tqdm(total=len(jobs), desc="embed") as bar:
    while True:
        item = work.get()
        if item is None:
            break
        person, rec, img = item
        if img is None:
            stats["failed"] += 1
        else:
            found = faces_in(img)
            if found:
                results[person].append({**rec, "faces": found})
                stats["ok"] += 1
                stats["faces"] += len(found)
            else:
                stats["no_face"] += 1
        pending[person] -= 1
        if pending[person] == 0:
            flush(person)
        bar.update(1)

for person in list(results):      # anyone with zero candidate images
    flush(person)

print(stats)


# ==== CELL 6 - summary =====================================================
import glob

paths = sorted(glob.glob(f"{OUT_DIR}/*.json"))
counts, empty = [], 0
for path in paths:
    with open(path, encoding="utf-8") as fh:
        d = json.load(fh)
    n = sum(len(im["faces"]) for im in d["images"])
    counts.append(n)
    empty += n == 0

print(f"{len(paths)} people written, {empty} with no usable face")
if counts:
    ordered = sorted(counts)
    print(f"faces per person: min={ordered[0]} "
          f"median={ordered[len(ordered) // 2]} max={ordered[-1]}")
    print(f"{sum(counts)} faces total")
