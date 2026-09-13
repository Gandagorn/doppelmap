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
DISCOVERY_CACHE = f"{WORK}/discovery_cache.json"
os.makedirs(OUT_DIR, exist_ok=True)

# Wikimedia blocks generic user agents. A descriptive one with contact info is
# required by their policy, and is the difference between this working and
# getting 403ed a few thousand requests in.
USER_AGENT = "doppelmap/1.0 (https://github.com/Gandagorn/doppelmap)"

MAX_IMAGES_PER_PERSON = 40
MAX_FACES_PER_IMAGE = 4       # group shots: keep the largest few, not all
MIN_DET_SCORE = 0.60          # detector confidence floor
MIN_FACE_PX = 60              # faces smaller than this embed poorly
DOWNLOAD_WORKERS = 16         # image fetches from upload.wikimedia.org
# The *API* is rate limited far more tightly than the image servers, and
# Wikimedia asks for serial requests against it. Sixteen threads hammering
# search got the whole run 429ed, which silently produced empty results for
# people like Bruce Willis and Dua Lipa. Keep this low.
DISCOVERY_WORKERS = 3
API_DELAY = 0.35              # seconds between API calls, per thread
QUEUE_DEPTH = 64              # decoded images awaiting the GPU (RAM bound)
REQUEST_TIMEOUT = 20          # seconds per HTTP request
MAX_RETRIES = 5               # image downloads, incl. 429 backoff
API_RETRIES = 6               # API calls, which hit 429s and need patience
MAX_IMAGE_BYTES = 25_000_000  # skip absurd originals
# Download a resized copy, not the original. Commons originals average
# 3.5 MB and reach 30 MB, which is ~480 GB across a full run and turns a
# two-hour job into a two-day one. Faces only need enough pixels to clear
# MIN_FACE_PX, so this loses nothing that matters.
THUMB_WIDTH = 800

print("output ->", OUT_DIR)


# ==== CELL 3 - people list and name matching ===============================
import json
import unicodedata

with open(PEOPLE_FILE, encoding="utf-8") as fh:
    raw_people = json.load(fh)

# Accepts the shapes the ranking step has produced so far: a bare list of
# names, a list of {"name"/"article", "views"} objects, or either of those
# wrapped in {"people": [...]}.
if isinstance(raw_people, dict):
    raw_people = raw_people.get("people", raw_people.get("names", []))

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


def names_the_person(name, title, url=""):
    """Does this Commons title or URL name the person?

    Recorded against every image rather than used to discard any, because
    this is the strongest identity signal available: Commons filenames are
    descriptive and human-curated. Downstream the named images become the
    anchor that decides *which* face in a gallery is this person, which
    matters because galleries are rarely clean -- a "Colin Hanks" search
    returned five Michael Cera photos from one Flickr set against two real
    ones, and the face recurring most across the whole gallery is Cera's.

    Unnamed images are still collected and still usable; they just do not
    get a vote on who the person is.

    Matches on the longest token, usually the surname. Last-token breaks on
    regnal names (Charles III -> "iii"), and any-token would let a Tom
    Holland photo satisfy a Tom Cruise query.
    """
    tokens = name_tokens(name)
    if not tokens:
        return True
    key = max(tokens, key=len)
    return key in strip_accents(f"{title} {url}").replace("_", " ").lower()


# ==== CELL 3A - shared HTTP helpers ========================================
# Split out of the discovery cell so the people filter below can use them
# too. Order matters: the filter has to shrink PEOPLE *before* CELL 4
# builds its work list from it, or it silently does nothing.
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
    """GET with exponential backoff. Returns None only after giving up.

    None means "we do not know", never "there is nothing" -- the caller has
    to keep those apart, because a rate-limited answer recorded as an empty
    result would be cached permanently by the resume logic.
    """
    for attempt in range(API_RETRIES):
        try:
            time.sleep(API_DELAY)
            r = session().get(url, params=params, timeout=REQUEST_TIMEOUT)
            if r.status_code == 429:
                # Honour Retry-After when given; otherwise back off hard.
                wait = float(r.headers.get("Retry-After") or 0) or min(60, 5 * 2 ** attempt)
                print(f"  rate limited, waiting {wait:.0f}s")
                time.sleep(wait)
                continue
            if r.ok:
                return r.json()
            if 500 <= r.status_code < 600:
                time.sleep(min(60, 5 * 2 ** attempt))
                continue
            return None                        # 4xx other than 429: real error
        except requests.RequestException:
            time.sleep(min(30, 2 * 2 ** attempt))
    return None


# ==== CELL 3B - keep only real people ======================================
# Pageview rankings are full of fictional characters: Wade Wilson, Sherlock
# Holmes, Darth Vader all out-rank plenty of real actors. Their Commons
# images are drawings, costumes and stills, so they poison a face map.
#
# Wikidata settles it authoritatively -- "instance of (P31) = human (Q5)".
# That keeps historical figures like Cleopatra and Napoleon, who a
# born-after-photography heuristic would wrongly discard, and drops
# fictional humans like Sherlock Holmes, who a name check never could.
#
# Two batched calls per 50 people, on endpoints unrelated to the heavily
# rate-limited Commons search, so this is cheap: ~200 requests for 5,000.

REAL_PEOPLE_CACHE = f"{WORK}/real_people.json"
HUMAN = "Q5"


def _chunks(seq, n=50):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def wikidata_ids(names):
    """Wikipedia titles -> Wikidata QIDs, following redirects."""
    out = {}
    for batch in _chunks(names):
        data = api_get("https://en.wikipedia.org/w/api.php", {
            "action": "query", "format": "json", "prop": "pageprops",
            "redirects": "1", "titles": "|".join(batch),
        })
        if not data or "query" not in data:
            continue
        q = data["query"]
        # Map the title the API answers with back to the name we asked for.
        norm = {n["from"]: n["to"] for n in q.get("normalized", [])}
        redir = {r["from"]: r["to"] for r in q.get("redirects", [])}
        asked = {}
        for name in batch:
            t = norm.get(name, name)
            asked[redir.get(t, t)] = name
        for page in q.get("pages", {}).values():
            qid = (page.get("pageprops") or {}).get("wikibase_item")
            if qid:
                out[asked.get(page["title"], page["title"])] = qid
    return out


def humans_only(names):
    """Subset of `names` that Wikidata says are actual people."""
    cached = {}
    if os.path.exists(REAL_PEOPLE_CACHE):
        with open(REAL_PEOPLE_CACHE, encoding="utf-8") as fh:
            cached = json.load(fh)

    unknown = [n for n in names if n not in cached]
    if unknown:
        ids = wikidata_ids(unknown)
        for batch in _chunks([n for n in unknown if n in ids]):
            data = api_get("https://www.wikidata.org/w/api.php", {
                "action": "wbgetentities", "format": "json",
                "ids": "|".join(ids[n] for n in batch), "props": "claims",
            })
            if not data:
                continue
            for name in batch:
                entity = (data.get("entities") or {}).get(ids[name])
                if entity is None:
                    continue
                claims = entity.get("claims", {}).get("P31", [])
                kinds = [c["mainsnak"]["datavalue"]["value"]["id"]
                         for c in claims if c["mainsnak"].get("datavalue")]
                cached[name] = HUMAN in kinds
        # Anything still unresolved is kept rather than guessed away: a
        # missing Wikidata entry is not evidence of being fictional.
        for name in unknown:
            cached.setdefault(name, True)
        tmp = REAL_PEOPLE_CACHE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(cached, fh, ensure_ascii=False)
        os.replace(tmp, REAL_PEOPLE_CACHE)

    return [n for n in names if cached.get(n, True)]


before = len(PEOPLE)
PEOPLE = humans_only(PEOPLE)
print(f"{before - len(PEOPLE)} non-people removed, {len(PEOPLE)} remain")


# ==== CELL 4 - discover candidate images ===================================
# Needs CELL 3A (session/api_get) and CELL 3B (filtered PEOPLE).
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
        return None            # unknown, not empty -- see api_get

    out = []
    for page in (data.get("query", {}).get("pages") or {}).values():
        info = (page.get("imageinfo") or [{}])[0]
        title = page.get("title", "")
        if not info.get("url") or not str(info.get("mime", "")).startswith("image/"):
            continue
        if (info.get("size") or 0) > MAX_IMAGE_BYTES:
            continue
        meta = info.get("extmetadata") or {}
        out.append({
            "title": title,
            # Whether the filename names this person -- the anchor signal
            # the prototype step uses to pick their face out of a gallery.
            "names_person": names_the_person(name, title, info["url"]),
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

# Cached separately from the per-person outputs: discovery is a full pass
# over the Commons API for everyone, and it would otherwise be repeated in
# full on every re-run -- including ones that only died partway through
# downloading. Keeping it also means a settings change that only affects
# the download/embed stage costs no API traffic at all.
cache = {}
if os.path.exists(DISCOVERY_CACHE):
    with open(DISCOVERY_CACHE, encoding="utf-8") as fh:
        cache = json.load(fh)
    print(f"{len(cache)} people already discovered (cached)")

missing = [p for p in todo if p not in cache]
failed = []
if missing:
    with ThreadPoolExecutor(max_workers=DISCOVERY_WORKERS) as pool:
        for person, found in zip(
            missing, tqdm(pool.map(discover, missing), total=len(missing), desc="discover")
        ):
            # Only a real answer is cached. A failure stays uncached so the
            # next run retries the person instead of inheriting a wrong
            # "they have no images" verdict forever.
            if found is None:
                failed.append(person)
            else:
                cache[person] = found
    tmp = DISCOVERY_CACHE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(cache, fh, ensure_ascii=False)
    os.replace(tmp, DISCOVERY_CACHE)

if failed:
    print(f"WARNING: discovery failed for {len(failed)} people "
          f"(rate limiting?) -- they are skipped this run and retried next. "
          f"e.g. {failed[:5]}")

# Only people we actually have an answer for get processed, so nobody is
# written out empty on the strength of a failed lookup.
candidates = {p: cache[p] for p in todo if p in cache}
print(f"{sum(len(v) for v in candidates.values())} candidate images "
      f"across {len(candidates)} people")


# ==== CELL 5 - download (threads) + embed (GPU) ============================
import queue
import urllib.parse

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from insightface.utils import face_align

app = FaceAnalysis(
    name="buffalo_l",
    allowed_modules=["detection", "recognition"],
    providers=["CUDAExecutionProvider"],
)
app.prepare(ctx_id=0, det_size=(640, 640))
recognizer = app.models["recognition"]


def embed_with_flip(img, face):
    """Embedding averaged over the face and its mirror image.

    Test-time flip augmentation. ArcFace is sensitive to which way a head
    is turned and to light falling on one side, and averaging a face with
    its mirror cancels part of that, giving a slightly steadier vector for
    the same person across photos.

    Costs one extra forward pass on a 112x112 crop, which is nothing next
    to detection -- both views go through as a single batch.
    """
    aligned = face_align.norm_crop(img, landmark=face.kps, image_size=112)
    feats = np.asarray(recognizer.get_feat([aligned, cv2.flip(aligned, 1)]))
    summed = feats.reshape(2, -1).sum(axis=0)
    return summed / np.linalg.norm(summed)


def thumb_url(rec, width=THUMB_WIDTH):
    """A resized copy of a Commons image, via Special:FilePath.

    Built from the title, which every cached record already has, so
    switching to thumbnails costs no re-discovery. Deriving the
    upload.wikimedia.org /thumb/ path directly does not work: the allowed
    widths are a per-image bucket list and anything else returns HTTP 400.
    Special:FilePath negotiates that server-side and redirects to the
    nearest valid size.
    """
    title = rec.get("title", "")
    if ":" not in title:
        return rec["original_url"]
    name = urllib.parse.quote(title.split(":", 1)[1].replace(" ", "_"))
    return f"https://commons.wikimedia.org/wiki/Special:FilePath/{name}?width={width}"


def fetch(job):
    """Download and decode one image. Runs on a download thread.

    Handles 429 properly. The first version treated a rate-limited reply as
    an ordinary failure and gave up after three one-second retries, so under
    throttling most images were quietly dropped and people ended up with a
    fraction of their photos -- or none at all, which then looked exactly
    like "this person has no usable pictures".
    """
    person, rec = job
    for attempt in range(MAX_RETRIES):
        try:
            url = thumb_url(rec) if attempt < MAX_RETRIES - 1 else rec["original_url"]
            r = session().get(url, timeout=REQUEST_TIMEOUT)
            if r.status_code == 429:
                wait = float(r.headers.get("Retry-After") or 0) or min(60, 5 * 2 ** attempt)
                time.sleep(wait)
                continue
            if r.ok and r.content:
                img = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
                return person, rec, img
            if r.status_code < 500:
                return person, rec, None      # 404 etc: retrying will not help
        except requests.RequestException:
            pass
        time.sleep(min(30, 2 * 2 ** attempt))
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
        if f.kps is None:          # no landmarks, so no reliable alignment
            continue
        found.append({
            "bbox": [round(v, 1) for v in (x1, y1, x2, y2)],
            "det_score": round(float(f.det_score), 3),
            # Rounded to 4 decimals: measured on real embeddings that shifts
            # cosine similarity by at most 1.7e-04, far below anything that
            # could reorder a ranking, while cutting each stored vector from
            # ~13.7 KB to ~3.5 KB. Across 5,000 people that is the difference
            # between roughly 2 GB and 500 MB on Drive.
            "emb": [round(v, 4) for v in embed_with_flip(img, f).astype(float).tolist()],
        })
    found.sort(key=lambda d: -(d["bbox"][2] - d["bbox"][0]) * (d["bbox"][3] - d["bbox"][1]))
    return found[:MAX_FACES_PER_IMAGE]


jobs = [(p, rec) for p, recs in candidates.items() for rec in recs]
results = {p: [] for p in candidates}
pending = {p: len(recs) for p, recs in candidates.items()}
stats = {"ok": 0, "no_face": 0, "failed": 0, "faces": 0}
# `failed` counts downloads that never produced an image. A large value
# means throttling or dead URLs, not people without photos -- worth knowing
# before a thin dataset is mistaken for a thin subject.


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


def producer(out_queue, todo):
    """Download in parallel, feeding the GPU thread through `out_queue`.

    The queue and job list are arguments, not globals. They used to be
    looked up globally, so re-running this cell after an interrupt left the
    previous run's producer alive and writing its stale results into the
    new run's queue -- which surfaced as a KeyError on a person the new run
    had already finished and flushed.
    """
    with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as pool:
        for item in pool.map(fetch, todo):
            out_queue.put(item)   # blocks when full, throttling to GPU speed
    out_queue.put(None)


threading.Thread(target=producer, args=(work, jobs), daemon=True).start()

with tqdm(total=len(jobs), desc="embed") as bar:
    while True:
        item = work.get()
        if item is None:
            break
        person, rec, img = item
        if person not in pending:
            continue              # stale item from an earlier interrupted run
        if img is None:
            stats["failed"] += 1
        else:
            found = faces_in(img)
            if found:
                # The size actually detected on, which is NOT rec["width"].
                # Commons serves ?width=800 from a per-image bucket list and
                # picks the nearest available (960, 727, 509 ...), and the
                # last retry falls back to the full-size original -- so the
                # scale cannot be recovered later. Without this the boxes get
                # normalised against the original's dimensions and every face
                # in an image larger than the thumbnail lands up and to the
                # left of the real one.
                height, width = img.shape[:2]
                results[person].append(
                    {**rec, "det_width": width, "det_height": height, "faces": found}
                )
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


# ==== CELL 7 - one-off repair for data collected before det_width ==========
# Only needed for people collected by an OLDER version of CELL 5, which stored
# the original image's dimensions while measuring face boxes on a thumbnail.
# CELL 5 now records the detection size directly, so this is not needed again.
#
# Metadata only: no images are downloaded and nothing is re-embedded. Safe to
# re-run and safe to interrupt -- anyone already carrying det_width is skipped.

import glob, json, os

COMMONS_API = "https://commons.wikimedia.org/w/api.php"
BATCH = 50          # the API's limit for a titles= query


def thumb_sizes(titles):
    """{title: (width, height)} for each title's THUMB_WIDTH thumbnail.

    Commons serves ?width=800 from a per-image bucket list, so the answer is
    960 or 727 or 509 rather than 800 -- which is exactly why the scale
    cannot be reconstructed from the stored record and has to be asked for.
    """
    data = api_get(COMMONS_API, {
        "action": "query", "format": "json", "prop": "imageinfo",
        "iiprop": "url|size", "iiurlwidth": THUMB_WIDTH,
        "titles": "|".join(titles),
    })
    sizes = {}
    for page in (data or {}).get("query", {}).get("pages", {}).values():
        info = (page.get("imageinfo") or [{}])[0]
        # A file narrower than THUMB_WIDTH is served as-is and reports no
        # thumb dimensions, so fall back to its own size.
        width = info.get("thumbwidth") or info.get("width")
        height = info.get("thumbheight") or info.get("height")
        if width and height:
            sizes[page["title"]] = (int(width), int(height))
    return sizes


fixed = people = 0
for path in tqdm(sorted(glob.glob(f"{OUT_DIR}/*.json")), desc="repair"):
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    images = payload["images"] if isinstance(payload, dict) else payload
    todo = [r for r in images
            if r.get("faces") and not r.get("det_width") and r.get("title")]
    if not todo:
        continue

    sizes = {}
    for i in range(0, len(todo), BATCH):
        sizes.update(thumb_sizes([r["title"] for r in todo[i:i + BATCH]]))

    for rec in todo:
        size = sizes.get(rec["title"])
        if not size:
            continue                       # unknown: leave it for a re-run
        width, height = size
        # CELL 5's last retry falls back to the full-size original, so a box
        # reaching past the thumbnail was measured on the original instead.
        # Trust the boxes rather than the assumption.
        if (max(f["bbox"][2] for f in rec["faces"]) > width + 1
                or max(f["bbox"][3] for f in rec["faces"]) > height + 1):
            width = rec.get("width") or width
            height = rec.get("height") or height
        rec["det_width"], rec["det_height"] = int(width), int(height)
        fixed += 1

    tmp = path + ".tmp"                    # same temp-then-rename as flush()
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False)
    os.replace(tmp, path)
    people += 1

print(f"repaired {fixed} images across {people} people")
