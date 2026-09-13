"""Run the collector's cells locally, against the real Commons API, with
only the GPU-dependent parts stubbed.

Exercises for real: people loading, the Wikidata human filter, discovery
and its cache, thumbnail URL building, downloading, the producer/consumer
queue, per-person checkpointing, the resume path, and the summary.

Stubbed: cv2 (decode returns a fake frame), insightface (detection returns
a fixed number of synthetic faces). Everything those feed into still runs.
"""
import io
from pathlib import Path
import json
import os
import re
import shutil
import sys
import tempfile
import types

import numpy as np

SRC = str(Path(__file__).resolve().parents[2] / "data" / "collect_commons_faces.py")
FAKE_FACES_PER_IMAGE = 2


# ---------------------------------------------------------------- stubs ----
def install_stubs():
    cv2 = types.ModuleType("cv2")
    cv2.IMREAD_COLOR = 1
    # A decoded frame only has to be an array of plausible shape; the real
    # pixels are only ever read by the detector, which is stubbed too.
    cv2.imdecode = lambda buf, flag: (
        np.zeros((600, 800, 3), np.uint8) if buf is not None and len(buf) > 100 else None
    )
    cv2.flip = lambda img, code: img
    sys.modules["cv2"] = cv2

    face = types.SimpleNamespace()

    class FakeFace:
        def __init__(self, i):
            self.det_score = 0.9
            self.bbox = np.array([10.0 + i * 90, 20.0, 150.0 + i * 90, 190.0])
            self.kps = np.zeros((5, 2), np.float32)

    class FakeRec:
        def get_feat(self, imgs):
            rng = np.random.default_rng(len(imgs))
            return rng.normal(size=(len(imgs), 512)).astype(np.float32)

    class FakeApp:
        def __init__(self, **kw): self.models = {"recognition": FakeRec()}
        def prepare(self, **kw): pass
        def get(self, img): return [FakeFace(i) for i in range(FAKE_FACES_PER_IMAGE)]

    ia = types.ModuleType("insightface")
    app_mod = types.ModuleType("insightface.app")
    app_mod.FaceAnalysis = FakeApp
    utils_mod = types.ModuleType("insightface.utils")
    fa = types.ModuleType("insightface.utils.face_align")
    fa.norm_crop = lambda img, landmark, image_size=112: np.zeros((112, 112, 3), np.uint8)
    utils_mod.face_align = fa
    ia.app, ia.utils = app_mod, utils_mod
    sys.modules.update({"insightface": ia, "insightface.app": app_mod,
                        "insightface.utils": utils_mod,
                        "insightface.utils.face_align": fa})


def cells_of(path):
    src = io.open(path, encoding="utf-8").read()
    out = {}
    for chunk in re.split(r"\n(?=# ==== CELL )", src):
        m = re.search(r"==== CELL (\w+)", chunk)
        if m:
            # Colab magics and the Drive mount cannot run here.
            body = "\n".join(
                "" if (l.lstrip()[:1] in "!%" or "drive.mount" in l or "from google.colab" in l)
                else l
                for l in chunk.splitlines()
            )
            out[m.group(1)] = body
    return out


def run(label, code, g):
    try:
        exec(compile(code, f"<{label}>", "exec"), g)
        return True
    except Exception as e:
        import traceback
        print(f"\n!!! {label} FAILED: {type(e).__name__}: {e}")
        traceback.print_exc(limit=3)
        return False


def main():
    install_stubs()
    work = tempfile.mkdtemp(prefix="dm_validate_")
    cells = cells_of(SRC)

    # A deliberately awkward people list: real, fictional, and nonexistent,
    # wrapped the way the user's own file is.
    people = ["Tom Hanks", "Zendaya", "Colin Hanks", "Wade Wilson",
              "Darth Vader", "Zzqx Notarealperson"]
    io.open(os.path.join(work, "consistent_people.json"), "w", encoding="utf-8").write(
        json.dumps({"people": [{"name": n, "views": 100} for n in people]}))

    g = {"__name__": "__main__"}
    ok = True

    print("=" * 64)
    print("CELL 2  config")
    cell2 = cells["2"].replace('WORK = "/content/drive/MyDrive/Projects/doppelmap"',
                               f'WORK = {work!r}')
    cell2 = cell2.replace('PEOPLE_FILE = f"{WORK}/people_top5000.json"',
                          'PEOPLE_FILE = f"{WORK}/consistent_people.json"')
    ok &= run("CELL 2", cell2, g)
    print(f"   WORK={g.get('WORK')}\n   THUMB_WIDTH={g.get('THUMB_WIDTH')} "
          f"DISCOVERY_WORKERS={g.get('DISCOVERY_WORKERS')} DOWNLOAD_WORKERS={g.get('DOWNLOAD_WORKERS')}")

    for cid, title in [("3", "people list"), ("3A", "http helpers"),
                       ("3B", "human filter"), ("4", "discovery"),
                       ("5", "download + embed"), ("6", "summary")]:
        print("=" * 64)
        print(f"CELL {cid}  {title}")
        ok &= run(f"CELL {cid}", cells[cid], g)
        if not ok:
            break
        if cid == "3":
            print(f"   PEOPLE={g['PEOPLE']}")
        if cid == "3B":
            print(f"   after filter: {g['PEOPLE']}")
        if cid == "4":
            print(f"   candidates: "
                  f"{ {k: len(v) for k, v in g['candidates'].items()} }")
        if cid == "5":
            print(f"   stats: {g['stats']}")

    print("=" * 64)
    if not ok:
        print("PIPELINE FAILED")
        return 1

    # ------------------------------------------------------------ asserts --
    checks = []
    out_dir = g["OUT_DIR"]
    files = sorted(f for f in os.listdir(out_dir) if f.endswith(".json"))
    checks.append(("fictional people filtered out",
                   "Wade Wilson" not in g["PEOPLE"] and "Darth Vader" not in g["PEOPLE"]))
    checks.append(("real people kept",
                   {"Tom Hanks", "Zendaya", "Colin Hanks"} <= set(g["PEOPLE"])))
    checks.append(("unknown name kept (not guessed away)",
                   "Zzqx Notarealperson" in g["PEOPLE"]))
    checks.append(("discovery cache written",
                   os.path.exists(g["DISCOVERY_CACHE"])))
    checks.append(("human/not-human cache written",
                   os.path.exists(g["REAL_PEOPLE_CACHE"])))
    checks.append(("one output file per surviving person",
                   len(files) == len(g["PEOPLE"])))
    checks.append(("images were actually downloaded", g["stats"]["ok"] > 0))
    checks.append(("no download failures", g["stats"]["failed"] == 0))

    payloads = {f[:-5]: json.load(io.open(os.path.join(out_dir, f), encoding="utf-8"))
                for f in files}
    any_img = next((im for p in payloads.values() for im in p["images"]), None)
    checks.append(("records carry faces with bbox + emb",
                   bool(any_img) and "faces" in any_img
                   and {"bbox", "det_score", "emb"} <= set(any_img["faces"][0])))
    checks.append(("embedding is 512-d",
                   bool(any_img) and len(any_img["faces"][0]["emb"]) == 512))
    checks.append(("embedding is unit length",
                   bool(any_img) and abs(np.linalg.norm(any_img["faces"][0]["emb"]) - 1) < 1e-3))
    checks.append(("names_person recorded",
                   bool(any_img) and "names_person" in any_img))
    checks.append(("licence metadata preserved",
                   bool(any_img) and "license" in any_img and "creator" in any_img))
    checks.append(("views carried through",
                   all(p.get("views") == 100 for p in payloads.values())))
    checks.append(("no .tmp files left behind",
                   not any(f.endswith(".tmp") for f in os.listdir(out_dir))))

    # Resume: a second pass must do nothing at all.
    before = {f: os.path.getmtime(os.path.join(out_dir, f)) for f in files}
    run("CELL 4 rerun", cells["4"], g)
    checks.append(("resume skips everyone already collected", not g["candidates"]))
    after = {f: os.path.getmtime(os.path.join(out_dir, f)) for f in files}
    checks.append(("resume rewrote nothing", before == after))

    # Thumbnails: are we really downloading small copies?
    checks.append(("thumbnail URL used, not the original",
                   "Special:FilePath" in g["thumb_url"](any_img)))

    print("VALIDATION")
    for name, passed in checks:
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")
    failed = [n for n, p in checks if not p]
    print("\n" + ("ALL CHECKS PASS" if not failed else f"FAILED: {failed}"))

    print(f"\nper-person output ({len(files)} files):")
    for n, p in sorted(payloads.items()):
        nf = sum(len(i["faces"]) for i in p["images"])
        print(f"   {n:24} {len(p['images']):3} images, {nf:3} faces")

    shutil.rmtree(work, ignore_errors=True)
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
