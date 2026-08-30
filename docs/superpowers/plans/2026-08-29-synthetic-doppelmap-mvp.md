# Synthetic Doppelmap MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a synthetic-data generator that produces a `graph.json` (+ placeholder thumbnails) matching the real pipeline's eventual output contract, and a working V1 frontend that renders, searches, and navigates that graph — so the website can be built and demoed before the real face-embedding pipeline exists.

**Architecture:** Two independent artifacts joined by one JSON contract. A Python package (`code/similarity_map/pipeline/`) generates synthetic 512-d embeddings with cluster structure, builds a cosine-kNN similarity graph with a mutual-kNN filter, lays it out in 2D, and writes `graph.json` + initials-avatar thumbnails straight into the frontend's `public/data/` folder. A Vite + TypeScript SPA (`code/web/`) fetches that JSON, renders it with Sigma.js/graphology, and implements search, hover, click-to-inspect, and a similarity sidebar. Swapping in the real pipeline later only requires replacing the embedding-generation step — everything downstream (graph construction, layout, frontend) is written to be reusable as-is.

**Tech Stack:** Python (numpy, scikit-learn, umap-learn, Pillow, pytest) for the data generator; TypeScript + Vite + Vitest + Sigma.js v3 + graphology for the frontend.

## Global Constraints

- Celebrity names come from a static curated list bundled in the repo (no network calls, no dataset download) — decided over Wikidata/IMDB-WIKI to keep the generator offline and deterministic.
- `k = 8` nearest neighbors per node (matches LLD §2.3).
- Edge set for rendering = **mutual**-kNN only (`a ∈ kNN(b) AND b ∈ kNN(a)`); the sidebar's ranked "similar" list is the raw **directed** top-8, kept separate (LLD §2.3).
- Layout coordinates are normalized into `[0, 10000]²` (LLD §2.4).
- `graph.json` schema is exactly the one in LLD §2.5: `{meta: {version, count, k}, nodes: [{id, name, x, y, deg, thumb, attr}], edges: [[src, dst, weight]], similar: {"<id>": [[id, weight], ...]}}`.
- Thumbnails are `96×96` WebP, quality 80 (LLD §2.2) — since there are no real photos yet, thumbnails are deterministic colored-initials placeholders, and each node's `attr` field honestly reads `"Synthetic placeholder — no real photo"` rather than a fake attribution.
- Frontend stack is Vite + vanilla TypeScript, Sigma.js v3 + graphology, **no router, no state library** (LLD §3.1) — total UI state is `{selectedId, hoveredId}`.
- Search is client-side substring/prefix match, diacritic-normalized (`NFD`), debounced 100ms, top-8 dropdown (LLD §3.1).
- Full WebGL image-node rendering (thumbnail sprites drawn directly on the graph canvas) is **out of scope for this plan** — the sidebar's `<img>` tag is the only place a thumbnail renders. Swapping in image nodes later is a frontend-only follow-up.
- Python tests run with `pytest`; TypeScript tests run with `vitest`.

---

## File Structure

```
code/
  similarity_map/
    pytest.ini
    requirements.txt
    __init__.py
    pipeline/
      __init__.py
      names.py            # curated celebrity name list
      embeddings.py       # synthetic clustered 512-d embeddings
      graph.py            # kNN, mutual-kNN filter, directed similar lists
      layout.py           # UMAP + coordinate normalization
      thumbnails.py       # initials-avatar WebP generator
      build_dataset.py    # CLI orchestration -> graph.json + thumbs/
    tests/
      test_names.py
      test_embeddings.py
      test_graph.py
      test_layout.py
      test_thumbnails.py
      test_build_dataset.py
  web/
    package.json
    tsconfig.json
    vite.config.ts
    index.html
    src/
      types.ts
      graphData.ts        # fetch + graphology transform
      sigmaSetup.ts        # zoom -> display-mode logic
      search.ts            # pure name search/filter
      interactions.ts      # flyToNode, sidebar data, selection reducer
      main.ts               # wires everything into the page
      style.css
    tests/
      graphData.test.ts
      sigmaSetup.test.ts
      search.test.ts
      interactions.test.ts
    public/
      data/                 # generator writes graph.json + thumbs/ here
  README.md                 # run instructions for generator + frontend
```

---

## Task 1: Project scaffold + curated celebrity name list

**Files:**
- Create: `code/similarity_map/pytest.ini`
- Create: `code/similarity_map/requirements.txt`
- Create: `code/similarity_map/__init__.py`
- Create: `code/similarity_map/pipeline/__init__.py`
- Create: `code/similarity_map/pipeline/names.py`
- Test: `code/similarity_map/tests/test_names.py`

**Interfaces:**
- Produces: `CELEBRITY_NAMES: list[str]` — used by every later pipeline task.

- [ ] **Step 1: Initialize git (this directory isn't a repo yet)**

```bash
cd /c/Users/danie/projects/doppelmap
git init
```

Expected: `Initialized empty Git repository in ...`

- [ ] **Step 2: Create the package scaffold**

`code/similarity_map/__init__.py`:
```python
```
(empty file — makes `similarity_map` importable as a package)

`code/similarity_map/pipeline/__init__.py`:
```python
```
(empty file — makes `pipeline` importable as a package)

`code/similarity_map/pytest.ini`:
```ini
[pytest]
pythonpath = ..
testpaths = tests
```

`code/similarity_map/requirements.txt`:
```
numpy
scikit-learn
umap-learn
pillow>=10.1
pytest
```

- [ ] **Step 3: Write the curated name list**

`code/similarity_map/pipeline/names.py`:
```python
"""Static, curated list of real public figures used to label synthetic
nodes. No network calls, no dataset download — deterministic and offline.
Names are not tied to any category/cluster; the synthetic embeddings in
embeddings.py assign cluster membership independently of who these people
actually are.
"""

CELEBRITY_NAMES = [
    "Tom Hanks", "Meryl Streep", "Denzel Washington", "Scarlett Johansson",
    "Leonardo DiCaprio", "Viola Davis", "Brad Pitt", "Cate Blanchett",
    "Morgan Freeman", "Natalie Portman", "Will Smith", "Charlize Theron",
    "Robert Downey Jr.", "Emma Stone", "Idris Elba", "Nicole Kidman",
    "Samuel L. Jackson", "Jennifer Lawrence", "Chiwetel Ejiofor", "Anne Hathaway",
    "Daniel Kaluuya", "Saoirse Ronan", "Mahershala Ali", "Emma Watson",
    "John Boyega", "Zendaya", "Michael B. Jordan", "Florence Pugh",
    "Oscar Isaac", "Lupita Nyong'o", "Timothee Chalamet", "Awkwafina",
    "Rami Malek", "Michelle Yeoh", "Riz Ahmed", "Sandra Oh",
    "Dev Patel", "Constance Wu", "Sterling K. Brown", "Yalitza Aparicio",
    "Beyonce", "Adele", "Bruno Mars", "Rihanna",
    "Ed Sheeran", "Taylor Swift", "Kendrick Lamar", "Billie Eilish",
    "The Weeknd", "Ariana Grande", "Drake", "Lady Gaga",
    "Harry Styles", "Dua Lipa", "John Legend", "Alicia Keys",
    "Bad Bunny", "Shakira", "Usher", "Katy Perry",
    "Serena Williams", "LeBron James", "Cristiano Ronaldo", "Lionel Messi",
    "Simone Biles", "Usain Bolt", "Naomi Osaka", "Roger Federer",
    "Rafael Nadal", "Novak Djokovic", "Megan Rapinoe", "Stephen Curry",
    "Kevin Durant", "Venus Williams", "Tiger Woods", "Michael Phelps",
    "Katie Ledecky", "Shohei Ohtani", "Coco Gauff", "Kylian Mbappe",
    "Barack Obama", "Michelle Obama", "Angela Merkel", "Justin Trudeau",
    "Jacinda Ardern", "Nelson Mandela", "Kamala Harris", "Emmanuel Macron",
    "Ruth Bader Ginsburg", "Volodymyr Zelenskyy",
    "Elon Musk", "Bill Gates", "Steve Jobs", "Mark Zuckerberg",
    "Satya Nadella", "Sundar Pichai", "Jeff Bezos", "Sheryl Sandberg",
    "Tim Cook", "Indra Nooyi",
    "George Clooney", "Julia Roberts", "Keanu Reeves", "Halle Berry",
    "Hugh Jackman", "Nicole Beharie", "Chris Hemsworth", "Gal Gadot",
    "Ryan Reynolds", "Blake Lively", "Jason Momoa", "Zoe Saldana",
    "Chadwick Boseman", "Angela Bassett", "Tom Holland", "Anya Taylor-Joy",
    "Pedro Pascal", "Jodie Comer", "John Cena", "Priyanka Chopra",
    "Stevie Wonder", "Whitney Houston", "Freddie Mercury", "Aretha Franklin",
    "Elton John", "Celine Dion", "Bob Marley", "Mariah Carey",
    "Prince", "Madonna",
    "Muhammad Ali", "Pele", "Maria Sharapova", "Michael Jordan",
    "Babe Ruth", "Jesse Owens", "Wayne Gretzky", "Diego Maradona",
    "Billie Jean King", "Jackie Robinson",
    "Albert Einstein", "Marie Curie", "Martin Luther King Jr.", "Mahatma Gandhi",
    "Malala Yousafzai", "Oprah Winfrey", "David Attenborough", "Malcolm X",
    "Frida Kahlo", "Rosa Parks",
]
```

- [ ] **Step 4: Write the test**

`code/similarity_map/tests/test_names.py`:
```python
from similarity_map.pipeline.names import CELEBRITY_NAMES


def test_names_are_unique_and_nonempty():
    assert len(CELEBRITY_NAMES) >= 150
    assert len(CELEBRITY_NAMES) == len(set(CELEBRITY_NAMES))
    assert all(isinstance(n, str) and n.strip() for n in CELEBRITY_NAMES)
```

- [ ] **Step 5: Install deps and run the test**

```bash
cd code/similarity_map
python -m pip install -r requirements.txt
python -m pytest -v
```

Expected: `test_names.py::test_names_are_unique_and_nonempty PASSED`

- [ ] **Step 6: Commit**

```bash
cd /c/Users/danie/projects/doppelmap
git add code/similarity_map/pytest.ini code/similarity_map/requirements.txt \
  code/similarity_map/__init__.py code/similarity_map/pipeline/__init__.py \
  code/similarity_map/pipeline/names.py code/similarity_map/tests/test_names.py
git commit -m "feat: scaffold pipeline package with curated celebrity name list"
```

---

## Task 2: Synthetic clustered embeddings

**Files:**
- Create: `code/similarity_map/pipeline/embeddings.py`
- Test: `code/similarity_map/tests/test_embeddings.py`

**Interfaces:**
- Consumes: nothing (takes a plain `list[str]` of names).
- Produces: `generate_synthetic_embeddings(names, *, avg_cluster_size=8.0, noise_scale=0.35, seed=42) -> dict[str, np.ndarray]`, `EMBEDDING_DIM = 512`. Task 6 (`build_dataset.py`) calls this directly.

- [ ] **Step 1: Write the failing test**

`code/similarity_map/tests/test_embeddings.py`:
```python
import numpy as np
from similarity_map.pipeline.embeddings import generate_synthetic_embeddings, EMBEDDING_DIM


def test_embeddings_are_unit_normalized():
    names = [f"Person {i}" for i in range(20)]
    embeddings = generate_synthetic_embeddings(names, seed=1)
    for vec in embeddings.values():
        assert vec.shape == (EMBEDDING_DIM,)
        assert abs(np.linalg.norm(vec) - 1.0) < 1e-5


def test_embeddings_deterministic_for_fixed_seed():
    names = [f"Person {i}" for i in range(20)]
    a = generate_synthetic_embeddings(names, seed=7)
    b = generate_synthetic_embeddings(names, seed=7)
    for name in names:
        assert np.allclose(a[name], b[name])


def test_cluster_structure_creates_higher_within_cluster_similarity():
    names = [f"Person {i}" for i in range(60)]
    embeddings = generate_synthetic_embeddings(
        names, avg_cluster_size=6, noise_scale=0.2, seed=3
    )
    vecs = np.stack([embeddings[n] for n in names])
    sim = vecs @ vecs.T
    off_diag = sim[~np.eye(len(names), dtype=bool)]
    best_neighbor_sim = np.sort(sim, axis=1)[:, -2]  # exclude self (always 1.0)
    assert best_neighbor_sim.mean() > off_diag.mean() + 0.15
```

- [ ] **Step 2: Run it and confirm it fails on import**

```bash
cd code/similarity_map
python -m pytest tests/test_embeddings.py -v
```

Expected: `ModuleNotFoundError: No module named 'similarity_map.pipeline.embeddings'`

- [ ] **Step 3: Implement**

`code/similarity_map/pipeline/embeddings.py`:
```python
"""Synthetic 512-d ArcFace-shaped embeddings with cluster structure, for
prototyping the pipeline before real face embeddings exist.
"""
import numpy as np

EMBEDDING_DIM = 512


def generate_synthetic_embeddings(
    names: list[str],
    *,
    avg_cluster_size: float = 8.0,
    noise_scale: float = 0.35,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Assign each name to a random synthetic cluster, then draw a unit
    embedding near that cluster's center. Clusters are arbitrary (not tied
    to profession, gender, etc.) — they exist purely so the kNN graph has
    believable structure instead of pure noise.

    Returns name -> L2-normalized float32 vector of shape (EMBEDDING_DIM,).
    """
    rng = np.random.default_rng(seed)
    n = len(names)
    n_clusters = max(1, round(n / avg_cluster_size))
    cluster_centers = rng.normal(size=(n_clusters, EMBEDDING_DIM)).astype(np.float32)
    cluster_centers /= np.linalg.norm(cluster_centers, axis=1, keepdims=True)

    cluster_ids = rng.integers(0, n_clusters, size=n)

    embeddings = {}
    for name, cluster_id in zip(names, cluster_ids):
        center = cluster_centers[cluster_id]
        noise = rng.normal(size=EMBEDDING_DIM).astype(np.float32)
        noise /= np.linalg.norm(noise) + 1e-8  # unit direction, uniform on the D-sphere
        noise *= noise_scale  # fixed magnitude, independent of EMBEDDING_DIM
        vec = center + noise
        vec /= np.linalg.norm(vec)
        embeddings[name] = vec.astype(np.float32)
    return embeddings
```

**Note (post-implementation correction):** the original draft here added raw `rng.normal(scale=noise_scale, ...)` noise directly, but an unnormalized Gaussian in 512-d has expected norm ≈ `noise_scale * sqrt(512)` ≈ `22.6 * noise_scale` — this swamps the unit-norm cluster center instead of perturbing it, and fails the cluster-structure test above. The code above (normalize the noise direction, then scale to a fixed magnitude) is the corrected version, verified during Task 2's implementation and review.

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_embeddings.py -v
```

Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
cd /c/Users/danie/projects/doppelmap
git add code/similarity_map/pipeline/embeddings.py code/similarity_map/tests/test_embeddings.py
git commit -m "feat: generate synthetic clustered embeddings"
```

---

## Task 3: Similarity graph construction (kNN, mutual filter, directed similar list)

**Files:**
- Create: `code/similarity_map/pipeline/graph.py`
- Test: `code/similarity_map/tests/test_graph.py`

**Interfaces:**
- Consumes: `embeddings: np.ndarray` of shape `(n, 512)`, L2-normalized (from Task 2, stacked).
- Produces: `build_knn(embeddings, k) -> (neighbor_idx, sim)`; `mutual_knn_edges(neighbor_idx, sim) -> list[tuple[int, int, float]]`; `directed_similar_lists(neighbor_idx, sim) -> dict[int, list[list]]`; `mutual_degrees(edges, n) -> list[int]`. All consumed by Task 6.

- [ ] **Step 1: Write the failing test**

`code/similarity_map/tests/test_graph.py`:
```python
import numpy as np
from similarity_map.pipeline.graph import (
    build_knn,
    mutual_knn_edges,
    directed_similar_lists,
    mutual_degrees,
)


def _two_tight_clusters():
    cluster_a = np.array([
        [1.0, 0.0, 0.0],
        [0.99, 0.01, 0.0],
        [0.98, 0.0, 0.02],
    ])
    cluster_b = np.array([
        [0.0, 1.0, 0.0],
        [0.01, 0.99, 0.0],
        [0.0, 0.98, 0.02],
    ])
    embeddings = np.vstack([cluster_a, cluster_b])
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings.astype(np.float32)


def test_build_knn_excludes_self_and_sorts_descending():
    embeddings = _two_tight_clusters()
    idx, sim = build_knn(embeddings, k=2)
    assert idx.shape == (6, 2)
    for row in range(6):
        assert row not in idx[row].tolist()
        assert sim[row, 0] >= sim[row, 1]


def test_mutual_knn_only_connects_within_cluster():
    embeddings = _two_tight_clusters()
    idx, sim = build_knn(embeddings, k=2)
    edges = mutual_knn_edges(idx, sim)
    assert len(edges) > 0
    for a, b, w in edges:
        assert (a < 3) == (b < 3)
        assert 0.0 <= w <= 1.0


def test_directed_similar_lists_ranked_and_covers_every_node():
    embeddings = _two_tight_clusters()
    idx, sim = build_knn(embeddings, k=2)
    similar = directed_similar_lists(idx, sim)
    assert set(similar.keys()) == set(range(6))
    for ranked in similar.values():
        sims = [s for _, s in ranked]
        assert sims == sorted(sims, reverse=True)


def test_mutual_degrees_matches_edge_count():
    embeddings = _two_tight_clusters()
    idx, sim = build_knn(embeddings, k=2)
    edges = mutual_knn_edges(idx, sim)
    deg = mutual_degrees(edges, n=6)
    assert sum(deg) == 2 * len(edges)
```

- [ ] **Step 2: Run it and confirm it fails on import**

```bash
python -m pytest tests/test_graph.py -v
```

Expected: `ModuleNotFoundError: No module named 'similarity_map.pipeline.graph'`

- [ ] **Step 3: Implement**

`code/similarity_map/pipeline/graph.py`:
```python
"""kNN + mutual-kNN similarity graph construction over pre-computed
L2-normalized embeddings.
"""
import numpy as np
from sklearn.neighbors import NearestNeighbors


def build_knn(embeddings: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Cosine kNN over an (n, d) L2-normalized embedding matrix.

    Returns (neighbor_indices, similarities), each shape (n, k), excluding
    self-matches, sorted by descending similarity.
    """
    n = embeddings.shape[0]
    k_query = min(k + 1, n)
    nn = NearestNeighbors(n_neighbors=k_query, metric="cosine").fit(embeddings)
    dist, idx = nn.kneighbors(embeddings)
    sim = 1 - dist
    return idx[:, 1:], sim[:, 1:]


def mutual_knn_edges(
    neighbor_idx: np.ndarray, sim: np.ndarray
) -> list[tuple[int, int, float]]:
    """Keep edge (a, b) only if a is in b's kNN list AND b is in a's kNN
    list. Returns deduplicated undirected edges as (min_id, max_id, weight),
    sorted by (src, dst).
    """
    neighbor_sets = [set(row.tolist()) for row in neighbor_idx]
    edges = {}
    n = neighbor_idx.shape[0]
    for a in range(n):
        for pos, b in enumerate(neighbor_idx[a]):
            b = int(b)
            if a in neighbor_sets[b]:
                key = (min(a, b), max(a, b))
                edges[key] = round(float(sim[a, pos]), 3)
    return sorted((a, b, w) for (a, b), w in edges.items())


def directed_similar_lists(
    neighbor_idx: np.ndarray, sim: np.ndarray
) -> dict[int, list[list]]:
    """Per-node ranked top-k similar list (directed, not mutual-filtered) —
    used for the sidebar, which should show a full top-k even for nodes the
    mutual filter would otherwise isolate.
    """
    result = {}
    for i in range(neighbor_idx.shape[0]):
        result[i] = [[int(j), round(float(s), 3)] for j, s in zip(neighbor_idx[i], sim[i])]
    return result


def mutual_degrees(edges: list[tuple[int, int, float]], n: int) -> list[int]:
    deg = [0] * n
    for a, b, _ in edges:
        deg[a] += 1
        deg[b] += 1
    return deg
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_graph.py -v
```

Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
cd /c/Users/danie/projects/doppelmap
git add code/similarity_map/pipeline/graph.py code/similarity_map/tests/test_graph.py
git commit -m "feat: build mutual-kNN similarity graph from embeddings"
```

---

## Task 4: 2D layout (UMAP + normalization)

**Files:**
- Create: `code/similarity_map/pipeline/layout.py`
- Test: `code/similarity_map/tests/test_layout.py`

**Interfaces:**
- Consumes: `embeddings: np.ndarray` shape `(n, 512)` (from Task 2).
- Produces: `compute_layout(embeddings, *, n_neighbors=15, seed=42) -> np.ndarray` shape `(n, 2)`; `normalize_coords(xy, canvas_size=10000.0) -> np.ndarray`. Consumed by Task 6.

- [ ] **Step 1: Write the failing test**

`code/similarity_map/tests/test_layout.py`:
```python
import numpy as np
from similarity_map.pipeline.layout import compute_layout, normalize_coords


def test_compute_layout_shape():
    embeddings = np.random.default_rng(0).normal(size=(30, 512)).astype(np.float32)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    xy = compute_layout(embeddings, seed=1)
    assert xy.shape == (30, 2)


def test_normalize_coords_within_bounds():
    xy = np.array([[-5.0, 100.0], [20.0, -30.0], [0.0, 0.0]])
    normalized = normalize_coords(xy, canvas_size=10000.0)
    assert normalized.min() >= 0.0
    assert normalized.max() <= 10000.0
    assert np.isclose(normalized.max(), 10000.0)


def test_normalize_coords_handles_degenerate_single_point():
    xy = np.array([[3.0, 3.0]])
    normalized = normalize_coords(xy)
    assert normalized.shape == (1, 2)
    assert np.all(np.isfinite(normalized))
```

- [ ] **Step 2: Run it and confirm it fails on import**

```bash
python -m pytest tests/test_layout.py -v
```

Expected: `ModuleNotFoundError: No module named 'similarity_map.pipeline.layout'`

- [ ] **Step 3: Implement**

`code/similarity_map/pipeline/layout.py`:
```python
"""2-D layout for the similarity graph: UMAP for global structure,
normalized to a fixed canvas.
"""
import numpy as np
import umap


def compute_layout(
    embeddings: np.ndarray, *, n_neighbors: int = 15, seed: int = 42
) -> np.ndarray:
    """Returns an (n, 2) float array of raw UMAP coordinates."""
    n_neighbors = min(n_neighbors, max(2, embeddings.shape[0] - 1))
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=0.15,
        metric="cosine",
        random_state=seed,
    )
    return reducer.fit_transform(embeddings)


def normalize_coords(xy: np.ndarray, canvas_size: float = 10000.0) -> np.ndarray:
    """Scale/translate coordinates to fill [0, canvas_size]^2, preserving
    aspect ratio.
    """
    mins = xy.min(axis=0)
    maxs = xy.max(axis=0)
    span = maxs - mins
    span = np.where(span == 0, 1.0, span)  # degenerate axis (e.g. n=1)
    scale = canvas_size / span.max()
    return (xy - mins) * scale
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_layout.py -v
```

Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
cd /c/Users/danie/projects/doppelmap
git add code/similarity_map/pipeline/layout.py code/similarity_map/tests/test_layout.py
git commit -m "feat: compute UMAP layout with normalized canvas coordinates"
```

---

## Task 5: Placeholder thumbnail generator

**Files:**
- Create: `code/similarity_map/pipeline/thumbnails.py`
- Test: `code/similarity_map/tests/test_thumbnails.py`

**Interfaces:**
- Consumes: a person's `name: str` and an `out_path: Path`.
- Produces: `generate_thumbnail(name, out_path) -> None` (writes a 96x96 WebP file); `THUMB_SIZE = 96`. Consumed by Task 6.

- [ ] **Step 1: Write the failing test**

`code/similarity_map/tests/test_thumbnails.py`:
```python
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
```

- [ ] **Step 2: Run it and confirm it fails on import**

```bash
python -m pytest tests/test_thumbnails.py -v
```

Expected: `ModuleNotFoundError: No module named 'similarity_map.pipeline.thumbnails'`

- [ ] **Step 3: Implement**

`code/similarity_map/pipeline/thumbnails.py`:
```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_thumbnails.py -v
```

Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
cd /c/Users/danie/projects/doppelmap
git add code/similarity_map/pipeline/thumbnails.py code/similarity_map/tests/test_thumbnails.py
git commit -m "feat: generate placeholder initials-avatar thumbnails"
```

---

## Task 6: Dataset assembly CLI (graph.json + thumbs/)

**Files:**
- Create: `code/similarity_map/pipeline/build_dataset.py`
- Test: `code/similarity_map/tests/test_build_dataset.py`

**Interfaces:**
- Consumes: `CELEBRITY_NAMES` (Task 1), `generate_synthetic_embeddings` (Task 2), `build_knn`/`mutual_knn_edges`/`directed_similar_lists`/`mutual_degrees` (Task 3), `compute_layout`/`normalize_coords` (Task 4), `generate_thumbnail` (Task 5).
- Produces: `build_dataset(*, count, k, seed, out_dir) -> dict` (the assembled graph, also written to `out_dir/graph.json`); a `main()` CLI entrypoint. This is the final pipeline artifact — Task 7 consumes its output (`graph.json` + `thumbs/`) from the frontend.

- [ ] **Step 1: Write the failing test**

`code/similarity_map/tests/test_build_dataset.py`:
```python
import json

import pytest

from similarity_map.pipeline.build_dataset import build_dataset


def test_build_dataset_end_to_end(tmp_path):
    graph = build_dataset(count=12, k=4, seed=1, out_dir=tmp_path)

    assert graph["meta"]["count"] == 12
    assert graph["meta"]["k"] == 4
    assert len(graph["nodes"]) == 12

    graph_json_path = tmp_path / "graph.json"
    assert graph_json_path.exists()
    on_disk = json.loads(graph_json_path.read_text(encoding="utf-8"))
    assert on_disk == graph

    for node in graph["nodes"]:
        assert set(node.keys()) == {"id", "name", "x", "y", "deg", "thumb", "attr"}
        thumb_path = tmp_path / node["thumb"]
        assert thumb_path.exists()

    for a, b, w in graph["edges"]:
        assert 0 <= a < 12
        assert 0 <= b < 12
        assert 0.0 <= w <= 1.0

    assert set(graph["similar"].keys()) == {str(i) for i in range(12)}


def test_build_dataset_rejects_count_over_available_names():
    with pytest.raises(ValueError):
        build_dataset(count=100_000, k=4, seed=1, out_dir=None)
```

- [ ] **Step 2: Run it and confirm it fails on import**

```bash
python -m pytest tests/test_build_dataset.py -v
```

Expected: `ModuleNotFoundError: No module named 'similarity_map.pipeline.build_dataset'`

- [ ] **Step 3: Implement**

`code/similarity_map/pipeline/build_dataset.py`:
```python
"""CLI: generate a fully synthetic graph.json + thumbs/ matching the real
doppelmap pipeline's output schema (doppelmap-lld.md section 2.5), using
synthetic embeddings instead of real face embeddings so the frontend can be
built and tested before the real pipeline exists.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from .names import CELEBRITY_NAMES
from .embeddings import generate_synthetic_embeddings
from .graph import build_knn, mutual_knn_edges, directed_similar_lists, mutual_degrees
from .layout import compute_layout, normalize_coords
from .thumbnails import generate_thumbnail

DEFAULT_OUT = Path(__file__).resolve().parents[2] / "web" / "public" / "data"


def build_dataset(*, count: int, k: int, seed: int, out_dir: Path) -> dict:
    names = CELEBRITY_NAMES[:count]
    if len(names) < count:
        raise ValueError(
            f"requested count={count} but only {len(names)} curated names are available"
        )

    embeddings_by_name = generate_synthetic_embeddings(names, seed=seed)
    embeddings = np.stack([embeddings_by_name[n] for n in names])

    neighbor_idx, sim = build_knn(embeddings, k=k)
    edges = mutual_knn_edges(neighbor_idx, sim)
    similar = directed_similar_lists(neighbor_idx, sim)
    deg = mutual_degrees(edges, n=len(names))

    xy = normalize_coords(compute_layout(embeddings, seed=seed))

    out_dir.mkdir(parents=True, exist_ok=True)
    thumbs_dir = out_dir / "thumbs"
    nodes = []
    for i, name in enumerate(names):
        thumb_rel = f"thumbs/{i}.webp"
        generate_thumbnail(name, thumbs_dir / f"{i}.webp")
        nodes.append({
            "id": i,
            "name": name,
            "x": round(float(xy[i, 0]), 1),
            "y": round(float(xy[i, 1]), 1),
            "deg": deg[i],
            "thumb": thumb_rel,
            "attr": "Synthetic placeholder — no real photo",
        })

    graph = {
        "meta": {"version": "synthetic-2026-08-29", "count": len(names), "k": k},
        "nodes": nodes,
        "edges": [[a, b, w] for a, b, w in edges],
        "similar": {str(i): ranked for i, ranked in similar.items()},
    }

    (out_dir / "graph.json").write_text(json.dumps(graph), encoding="utf-8")
    return graph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=150, help="number of synthetic celebrities")
    parser.add_argument("--k", type=int, default=8, help="kNN neighbors per node")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    graph = build_dataset(count=args.count, k=args.k, seed=args.seed, out_dir=args.out)
    print(f"Wrote {len(graph['nodes'])} nodes, {len(graph['edges'])} edges to {args.out}")


if __name__ == "__main__":
    main()
```

Note: `build_dataset` raises `ValueError` before touching `out_dir`, so `test_build_dataset_rejects_count_over_available_names` passing `out_dir=None` never reaches `out_dir.mkdir()`.

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_build_dataset.py -v
```

Expected: 2 passed

- [ ] **Step 5: Run the full pipeline test suite**

```bash
python -m pytest -v
```

Expected: all tests across test_names/test_embeddings/test_graph/test_layout/test_thumbnails/test_build_dataset pass.

- [ ] **Step 6: Generate the real dataset the frontend will consume**

```bash
python -m pipeline.build_dataset --count 150 --k 8 --seed 42
```

Expected output: `Wrote 150 nodes, <N> edges to .../code/web/public/data`, and `code/web/public/data/graph.json` + `code/web/public/data/thumbs/*.webp` (150 files) now exist.

- [ ] **Step 7: Commit**

```bash
cd /c/Users/danie/projects/doppelmap
git add code/similarity_map/pipeline/build_dataset.py code/similarity_map/tests/test_build_dataset.py \
  code/web/public/data
git commit -m "feat: assemble synthetic graph.json + thumbnails via CLI"
```

---

## Task 7: Frontend scaffold (Vite + TS + Vitest) + graph data loader

**Files:**
- Create: `code/web/package.json`
- Create: `code/web/tsconfig.json`
- Create: `code/web/vite.config.ts`
- Create: `code/web/index.html`
- Create: `code/web/src/types.ts`
- Create: `code/web/src/graphData.ts`
- Create: `code/web/src/style.css`
- Create: `code/web/src/main.ts`
- Test: `code/web/tests/graphData.test.ts`

**Interfaces:**
- Consumes: `graph.json` at `code/web/public/data/graph.json` (produced by Task 6).
- Produces: `GraphNode`, `GraphEdge`, `GraphData` types in `types.ts`; `loadGraphData(url) -> Promise<GraphData>` and `buildGraphology(data) -> Graph` in `graphData.ts`. Every later frontend task imports from `types.ts` and `graphData.ts`.

- [ ] **Step 1: Scaffold the Vite project files**

`code/web/package.json`:
```json
{
  "name": "doppelmap-web",
  "private": true,
  "version": "0.0.1",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "preview": "vite preview",
    "test": "vitest run"
  },
  "dependencies": {
    "graphology": "^0.25.4",
    "graphology-types": "^0.24.7",
    "sigma": "^3.0.0"
  },
  "devDependencies": {
    "typescript": "^5.5.4",
    "vite": "^5.4.0",
    "vitest": "^2.0.5"
  }
}
```

`code/web/tsconfig.json`:
```json
{
  "compilerOptions": {
    "target": "ES2020",
    "module": "ESNext",
    "moduleResolution": "Bundler",
    "strict": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "forceConsistentCasingInFileNames": true,
    "types": ["vite/client"]
  },
  "include": ["src", "tests"]
}
```

`code/web/vite.config.ts`:
```ts
import { defineConfig } from "vite";

export default defineConfig({
  root: ".",
  publicDir: "public",
  build: {
    outDir: "dist",
  },
});
```

`code/web/index.html`:
```html
<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Doppelmap</title>
    <link rel="stylesheet" href="/src/style.css" />
  </head>
  <body>
    <div id="app">
      <input id="search" type="text" placeholder="Search a name..." autocomplete="off" />
      <div id="search-results"></div>
      <div id="graph-container"></div>
      <aside id="sidebar" hidden></aside>
    </div>
    <script type="module" src="/src/main.ts"></script>
  </body>
</html>
```

`code/web/src/style.css`:
```css
:root {
  color-scheme: light dark;
}
body {
  margin: 0;
  font-family: system-ui, sans-serif;
}
#graph-container {
  position: fixed;
  inset: 0;
}
#search {
  position: fixed;
  top: 12px;
  left: 12px;
  z-index: 10;
  padding: 8px 12px;
  border-radius: 6px;
  border: 1px solid #8888;
  width: 240px;
}
#search-results {
  position: fixed;
  top: 48px;
  left: 12px;
  z-index: 10;
  width: 240px;
}
.search-result {
  padding: 6px 10px;
  background: canvas;
  cursor: pointer;
}
.search-result:hover {
  background: color-mix(in srgb, canvastext 10%, canvas);
}
#sidebar {
  position: fixed;
  top: 0;
  right: 0;
  bottom: 0;
  width: 320px;
  background: canvas;
  box-shadow: -2px 0 8px rgba(0, 0, 0, 0.2);
  padding: 16px;
  overflow-y: auto;
}
#sidebar[hidden] {
  display: none;
}
.similar-list li {
  cursor: pointer;
  padding: 4px 0;
}
@media (max-width: 640px) {
  #sidebar {
    top: auto;
    left: 0;
    right: 0;
    width: auto;
    height: 40vh;
  }
}
```

- [ ] **Step 2: Write shared types**

`code/web/src/types.ts`:
```ts
export interface GraphNode {
  id: number;
  name: string;
  x: number;
  y: number;
  deg: number;
  thumb: string;
  attr: string;
}

export type GraphEdge = [number, number, number];

export interface GraphData {
  meta: { version: string; count: number; k: number };
  nodes: GraphNode[];
  edges: GraphEdge[];
  similar: Record<string, [number, number][]>;
}
```

- [ ] **Step 3: Write the failing test for the data loader**

`code/web/tests/graphData.test.ts`:
```ts
import { describe, expect, it } from "vitest";
import { buildGraphology } from "../src/graphData";
import type { GraphData } from "../src/types";

function sampleData(): GraphData {
  return {
    meta: { version: "test", count: 3, k: 2 },
    nodes: [
      { id: 0, name: "Alice", x: 0, y: 0, deg: 2, thumb: "thumbs/0.webp", attr: "synthetic" },
      { id: 1, name: "Bob", x: 10, y: 10, deg: 1, thumb: "thumbs/1.webp", attr: "synthetic" },
      { id: 2, name: "Carol", x: 20, y: 5, deg: 1, thumb: "thumbs/2.webp", attr: "synthetic" },
    ],
    edges: [
      [0, 1, 0.9],
      [0, 2, 0.5],
    ],
    similar: {
      "0": [[1, 0.9], [2, 0.5]],
      "1": [[0, 0.9]],
      "2": [[0, 0.5]],
    },
  };
}

describe("buildGraphology", () => {
  it("creates one graph node per data node", () => {
    const graph = buildGraphology(sampleData());
    expect(graph.order).toBe(3);
  });

  it("creates one graph edge per data edge", () => {
    const graph = buildGraphology(sampleData());
    expect(graph.size).toBe(2);
  });

  it("copies node attributes across", () => {
    const graph = buildGraphology(sampleData());
    expect(graph.getNodeAttribute("1", "label")).toBe("Bob");
    expect(graph.getNodeAttribute("1", "thumb")).toBe("thumbs/1.webp");
  });

  it("connects the correct endpoints", () => {
    const graph = buildGraphology(sampleData());
    expect(graph.hasEdge("0", "1")).toBe(true);
    expect(graph.hasEdge("1", "2")).toBe(false);
  });
});
```

- [ ] **Step 4: Install deps and confirm the test fails on import**

```bash
cd code/web
npm install
npm test
```

Expected: fails — `src/graphData.ts` doesn't exist yet.

- [ ] **Step 5: Implement the loader**

`code/web/src/graphData.ts`:
```ts
import Graph from "graphology";
import type { GraphData } from "./types";

export function buildGraphology(data: GraphData): Graph {
  const graph = new Graph({ type: "undirected", multi: false });
  for (const node of data.nodes) {
    graph.addNode(String(node.id), {
      label: node.name,
      x: node.x,
      y: node.y,
      size: 3 + Math.sqrt(node.deg),
      thumb: node.thumb,
      attr: node.attr,
      deg: node.deg,
    });
  }
  for (const [a, b, weight] of data.edges) {
    graph.addEdge(String(a), String(b), { weight, size: 0.5 + weight });
  }
  return graph;
}

export async function loadGraphData(url: string): Promise<GraphData> {
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`failed to load graph data: ${res.status} ${res.statusText}`);
  }
  return (await res.json()) as GraphData;
}
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
npm test
```

Expected: 4 passed

- [ ] **Step 7: Write a minimal bootstrap and verify it loads real data**

`code/web/src/main.ts`:
```ts
import { loadGraphData, buildGraphology } from "./graphData";

async function bootstrap() {
  const data = await loadGraphData("/data/graph.json");
  const graph = buildGraphology(data);
  console.log(`loaded ${graph.order} nodes, ${graph.size} edges`);
}

bootstrap().catch((err) => {
  console.error(err);
});
```

```bash
npm run dev
```

Open the printed local URL in a browser, open devtools console, and confirm it logs `loaded 150 nodes, <N> edges` (using the `graph.json` generated in Task 6).

- [ ] **Step 8: Commit**

```bash
cd /c/Users/danie/projects/doppelmap
git add code/web/package.json code/web/package-lock.json code/web/tsconfig.json \
  code/web/vite.config.ts code/web/index.html code/web/src code/web/tests
git commit -m "feat: scaffold Vite/TS frontend with graph.json loader"
```

---

## Task 8: Sigma renderer with zoom-dependent labels

**Files:**
- Create: `code/web/src/sigmaSetup.ts`
- Modify: `code/web/src/main.ts`
- Test: `code/web/tests/sigmaSetup.test.ts`

**Interfaces:**
- Consumes: `buildGraphology`/`loadGraphData` (Task 7).
- Produces: `getDisplayMode(cameraRatio: number): "dot" | "label"` in `sigmaSetup.ts`, used again in Task 10's combined node reducer.

- [ ] **Step 1: Write the failing test**

`code/web/tests/sigmaSetup.test.ts`:
```ts
import { describe, expect, it } from "vitest";
import { getDisplayMode } from "../src/sigmaSetup";

describe("getDisplayMode", () => {
  it("shows plain dots when zoomed far out", () => {
    expect(getDisplayMode(1.0)).toBe("dot");
  });

  it("shows labels once zoomed past the threshold", () => {
    expect(getDisplayMode(0.3)).toBe("label");
  });

  it("treats the exact threshold as label mode", () => {
    expect(getDisplayMode(0.5)).toBe("label");
  });
});
```

- [ ] **Step 2: Run it and confirm it fails on import**

```bash
npm test
```

Expected: fails — `src/sigmaSetup.ts` doesn't exist yet.

- [ ] **Step 3: Implement**

`code/web/src/sigmaSetup.ts`:
```ts
export type DisplayMode = "dot" | "label";

const LABEL_THRESHOLD_RATIO = 0.5;

export function getDisplayMode(cameraRatio: number): DisplayMode {
  return cameraRatio <= LABEL_THRESHOLD_RATIO ? "label" : "dot";
}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
npm test
```

Expected: 3 passed (plus the 4 from Task 7 still passing)

- [ ] **Step 5: Wire Sigma into main.ts**

`code/web/src/main.ts`:
```ts
import Sigma from "sigma";
import { loadGraphData, buildGraphology } from "./graphData";
import { getDisplayMode } from "./sigmaSetup";

async function bootstrap() {
  const data = await loadGraphData("/data/graph.json");
  const graph = buildGraphology(data);

  const container = document.getElementById("graph-container");
  if (!container) throw new Error("#graph-container not found in DOM");

  const renderer = new Sigma(graph, container, {
    labelRenderedSizeThreshold: 0,
  });

  renderer.setSetting("nodeReducer", (_node, attrs) => {
    const mode = getDisplayMode(renderer.getCamera().ratio);
    return mode === "dot" ? { ...attrs, label: "" } : attrs;
  });
}

bootstrap().catch((err) => {
  console.error(err);
});
```

- [ ] **Step 6: Manually verify in the browser**

```bash
npm run dev
```

Open the app: confirm ~150 dots render at initial zoom, and that zooming in past the threshold reveals name labels, zooming back out hides them.

- [ ] **Step 7: Commit**

```bash
cd /c/Users/danie/projects/doppelmap
git add code/web/src/sigmaSetup.ts code/web/src/main.ts code/web/tests/sigmaSetup.test.ts
git commit -m "feat: render similarity graph with zoom-dependent labels"
```

---

## Task 9: Search bar

**Files:**
- Create: `code/web/src/search.ts`
- Create: `code/web/src/interactions.ts`
- Modify: `code/web/src/main.ts`
- Test: `code/web/tests/search.test.ts`

**Interfaces:**
- Consumes: `GraphNode`, `GraphData` types (Task 7).
- Produces: `searchNames(nodes, query, limit=8): GraphNode[]` in `search.ts`; `flyToNode(renderer, graph, nodeId, duration=500): void` in `interactions.ts` (extended further in Task 10).

- [ ] **Step 1: Write the failing test**

`code/web/tests/search.test.ts`:
```ts
import { describe, expect, it } from "vitest";
import { searchNames } from "../src/search";
import type { GraphNode } from "../src/types";

function node(id: number, name: string): GraphNode {
  return { id, name, x: 0, y: 0, deg: 0, thumb: "", attr: "" };
}

const NODES = [
  node(0, "Tom Hanks"),
  node(1, "Tom Holland"),
  node(2, "Beyonce"),
  node(3, "Bad Bunny"),
];

describe("searchNames", () => {
  it("returns empty array for blank query", () => {
    expect(searchNames(NODES, "  ")).toEqual([]);
  });

  it("ranks prefix matches before substring matches", () => {
    const results = searchNames(NODES, "tom");
    expect(results.map((n) => n.name)).toEqual(["Tom Hanks", "Tom Holland"]);
  });

  it("matches substrings not just prefixes", () => {
    const results = searchNames(NODES, "bunny");
    expect(results.map((n) => n.name)).toEqual(["Bad Bunny"]);
  });

  it("is case-insensitive and diacritic-insensitive", () => {
    const results = searchNames(NODES, "beyonce");
    expect(results.map((n) => n.name)).toEqual(["Beyonce"]);
  });

  it("respects the limit", () => {
    const manyNodes = Array.from({ length: 20 }, (_, i) => node(i, `Tom ${i}`));
    expect(searchNames(manyNodes, "tom", 5)).toHaveLength(5);
  });
});
```

- [ ] **Step 2: Run it and confirm it fails on import**

```bash
npm test
```

Expected: fails — `src/search.ts` doesn't exist yet.

- [ ] **Step 3: Implement search**

`code/web/src/search.ts`:
```ts
import type { GraphNode } from "./types";

function normalize(value: string): string {
  return value
    .normalize("NFD")
    .replace(/[̀-ͯ]/g, "")
    .toLowerCase();
}

export function searchNames(nodes: GraphNode[], query: string, limit = 8): GraphNode[] {
  const q = normalize(query.trim());
  if (!q) return [];

  const prefixMatches: GraphNode[] = [];
  const substringMatches: GraphNode[] = [];
  for (const node of nodes) {
    const normalizedName = normalize(node.name);
    if (normalizedName.startsWith(q)) {
      prefixMatches.push(node);
    } else if (normalizedName.includes(q)) {
      substringMatches.push(node);
    }
  }
  return [...prefixMatches, ...substringMatches].slice(0, limit);
}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
npm test
```

Expected: 5 passed (plus earlier suites still passing)

- [ ] **Step 5: Implement flyToNode**

`code/web/src/interactions.ts`:
```ts
import type Sigma from "sigma";
import type Graph from "graphology";

export function flyToNode(
  renderer: Sigma,
  graph: Graph,
  nodeId: string,
  duration = 500
): void {
  const x = graph.getNodeAttribute(nodeId, "x") as number;
  const y = graph.getNodeAttribute(nodeId, "y") as number;
  renderer.getCamera().animate({ x, y, ratio: 0.15 }, { duration });
}
```

- [ ] **Step 6: Wire the search bar into main.ts**

`code/web/src/main.ts`:
```ts
import Sigma from "sigma";
import { loadGraphData, buildGraphology } from "./graphData";
import { getDisplayMode } from "./sigmaSetup";
import { searchNames } from "./search";
import { flyToNode } from "./interactions";

async function bootstrap() {
  const data = await loadGraphData("/data/graph.json");
  const graph = buildGraphology(data);

  const container = document.getElementById("graph-container");
  if (!container) throw new Error("#graph-container not found in DOM");

  const renderer = new Sigma(graph, container, {
    labelRenderedSizeThreshold: 0,
  });

  renderer.setSetting("nodeReducer", (_node, attrs) => {
    const mode = getDisplayMode(renderer.getCamera().ratio);
    return mode === "dot" ? { ...attrs, label: "" } : attrs;
  });

  const searchInput = document.getElementById("search") as HTMLInputElement;
  const resultsEl = document.getElementById("search-results") as HTMLDivElement;

  let debounceHandle: ReturnType<typeof setTimeout> | undefined;
  searchInput.addEventListener("input", () => {
    clearTimeout(debounceHandle);
    debounceHandle = setTimeout(() => {
      const matches = searchNames(data.nodes, searchInput.value);
      resultsEl.innerHTML = "";
      for (const node of matches) {
        const item = document.createElement("div");
        item.className = "search-result";
        item.textContent = node.name;
        item.addEventListener("click", () => {
          flyToNode(renderer, graph, String(node.id));
          resultsEl.innerHTML = "";
          searchInput.value = node.name;
        });
        resultsEl.appendChild(item);
      }
    }, 100);
  });
}

bootstrap().catch((err) => {
  console.error(err);
});
```

- [ ] **Step 7: Manually verify in the browser**

```bash
npm run dev
```

Type a partial name into the search box; confirm a dropdown of matches appears, and clicking a result animates the camera to that node.

- [ ] **Step 8: Commit**

```bash
cd /c/Users/danie/projects/doppelmap
git add code/web/src/search.ts code/web/src/interactions.ts code/web/src/main.ts code/web/tests/search.test.ts
git commit -m "feat: add debounced name search with camera fly-to"
```

---

## Task 10: Hover highlighting, click-to-inspect sidebar, deselect

**Files:**
- Modify: `code/web/src/interactions.ts`
- Modify: `code/web/src/main.ts`
- Test: `code/web/tests/interactions.test.ts`

**Interfaces:**
- Consumes: `GraphData`, `GraphNode` types (Task 7); `flyToNode` (Task 9).
- Produces: `formatSimilarity(weight): string`, `getSidebarData(data, nodeId): SidebarData` in `interactions.ts`. Selection is just a plain `{selectedId, hoveredId}` object mutated directly in `main.ts` — no reducer/action layer, per the "keep it minimal" steer.

- [ ] **Step 1: Write the failing test**

`code/web/tests/interactions.test.ts`:
```ts
import { describe, expect, it } from "vitest";
import { formatSimilarity, getSidebarData } from "../src/interactions";
import type { GraphData } from "../src/types";

function sampleData(): GraphData {
  return {
    meta: { version: "test", count: 3, k: 2 },
    nodes: [
      { id: 0, name: "Alice", x: 0, y: 0, deg: 2, thumb: "thumbs/0.webp", attr: "synthetic" },
      { id: 1, name: "Bob", x: 10, y: 10, deg: 1, thumb: "thumbs/1.webp", attr: "synthetic" },
      { id: 2, name: "Carol", x: 20, y: 5, deg: 1, thumb: "thumbs/2.webp", attr: "synthetic" },
    ],
    edges: [
      [0, 1, 0.912],
      [0, 2, 0.5],
    ],
    similar: {
      "0": [[1, 0.912], [2, 0.5]],
      "1": [[0, 0.912]],
      "2": [[0, 0.5]],
    },
  };
}

describe("formatSimilarity", () => {
  it("formats a cosine weight as a rounded percentage", () => {
    expect(formatSimilarity(0.912)).toBe("91.2%");
    expect(formatSimilarity(0.5)).toBe("50%");
  });
});

describe("getSidebarData", () => {
  it("returns node info plus ranked similar list with formatted percentages", () => {
    const sidebar = getSidebarData(sampleData(), 0);
    expect(sidebar.name).toBe("Alice");
    expect(sidebar.similar).toEqual([
      { id: 1, name: "Bob", percent: "91.2%" },
      { id: 2, name: "Carol", percent: "50%" },
    ]);
  });

  it("throws for an unknown node id", () => {
    expect(() => getSidebarData(sampleData(), 99)).toThrow();
  });
});
```

- [ ] **Step 2: Run it and confirm it fails**

```bash
npm test
```

Expected: fails — `formatSimilarity` and `getSidebarData` don't exist yet.

- [ ] **Step 3: Extend interactions.ts**

`code/web/src/interactions.ts` (full file — adds to the `flyToNode` from Task 9):
```ts
import type Sigma from "sigma";
import type Graph from "graphology";
import type { GraphData } from "./types";

export function flyToNode(
  renderer: Sigma,
  graph: Graph,
  nodeId: string,
  duration = 500
): void {
  const x = graph.getNodeAttribute(nodeId, "x") as number;
  const y = graph.getNodeAttribute(nodeId, "y") as number;
  renderer.getCamera().animate({ x, y, ratio: 0.15 }, { duration });
}

export function formatSimilarity(weight: number): string {
  return `${Math.round(weight * 1000) / 10}%`;
}

export interface SidebarData {
  id: number;
  name: string;
  thumb: string;
  attr: string;
  similar: { id: number; name: string; percent: string }[];
}

export function getSidebarData(data: GraphData, nodeId: number): SidebarData {
  const node = data.nodes.find((n) => n.id === nodeId);
  if (!node) throw new Error(`node ${nodeId} not found`);
  const nodesById = new Map(data.nodes.map((n) => [n.id, n]));
  const ranked = data.similar[String(nodeId)] ?? [];
  return {
    id: node.id,
    name: node.name,
    thumb: node.thumb,
    attr: node.attr,
    similar: ranked.map(([id, weight]) => ({
      id,
      name: nodesById.get(id)?.name ?? "Unknown",
      percent: formatSimilarity(weight),
    })),
  };
}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
npm test
```

Expected: 3 passed in `interactions.test.ts` (plus all earlier suites still passing)

- [ ] **Step 5: Wire sidebar, hover, click, and Esc/background-deselect into main.ts**

`code/web/src/main.ts` (full file — replaces the Task 9 version's `nodeReducer` with a combined one, and adds sidebar/hover/click/deselect wiring):
```ts
import Sigma from "sigma";
import { loadGraphData, buildGraphology } from "./graphData";
import { getDisplayMode } from "./sigmaSetup";
import { searchNames } from "./search";
import { flyToNode, getSidebarData } from "./interactions";

async function bootstrap() {
  const data = await loadGraphData("/data/graph.json");
  const graph = buildGraphology(data);

  const container = document.getElementById("graph-container");
  if (!container) throw new Error("#graph-container not found in DOM");

  const renderer = new Sigma(graph, container, {
    labelRenderedSizeThreshold: 0,
  });

  // Total UI state: which node is selected (sidebar open) and which is
  // hovered (dims everything else). Just a plain object mutated in place —
  // two fields don't need a reducer.
  const selection: { selectedId: number | null; hoveredId: number | null } = {
    selectedId: null,
    hoveredId: null,
  };
  const sidebarEl = document.getElementById("sidebar") as HTMLElement;

  function renderSidebar() {
    if (selection.selectedId === null) {
      sidebarEl.hidden = true;
      sidebarEl.innerHTML = "";
      return;
    }
    const info = getSidebarData(data, selection.selectedId);
    sidebarEl.hidden = false;
    sidebarEl.innerHTML = `
      <img src="/data/${info.thumb}" width="96" height="96" alt="${info.name}" />
      <h2>${info.name}</h2>
      <p class="attr">${info.attr}</p>
      <ul class="similar-list">
        ${info.similar
          .map((s) => `<li data-id="${s.id}">${s.name} — ${s.percent}</li>`)
          .join("")}
      </ul>
    `;
    sidebarEl.querySelectorAll<HTMLLIElement>("li[data-id]").forEach((li) => {
      li.addEventListener("click", () => {
        selection.selectedId = Number(li.dataset.id);
        flyToNode(renderer, graph, String(selection.selectedId));
        renderSidebar();
      });
    });
  }

  renderer.on("clickNode", ({ node }) => {
    selection.selectedId = Number(node);
    flyToNode(renderer, graph, node);
    renderSidebar();
  });

  renderer.on("enterNode", ({ node }) => {
    selection.hoveredId = Number(node);
    renderer.refresh();
  });

  renderer.on("leaveNode", () => {
    selection.hoveredId = null;
    renderer.refresh();
  });

  renderer.on("clickStage", () => {
    selection.selectedId = null;
    renderSidebar();
  });

  window.addEventListener("keydown", (evt) => {
    if (evt.key === "Escape") {
      selection.selectedId = null;
      renderSidebar();
    }
  });

  renderer.setSetting("nodeReducer", (nodeId, attrs) => {
    const mode = getDisplayMode(renderer.getCamera().ratio);
    const display = { ...attrs };
    if (mode === "dot") display.label = "";

    if (selection.hoveredId !== null) {
      const hoveredKey = String(selection.hoveredId);
      const isHovered = nodeId === hoveredKey;
      const isNeighbor = graph.areNeighbors(nodeId, hoveredKey);
      if (!isHovered && !isNeighbor) {
        display.color = "#d8d8d8";
        display.label = "";
      }
    }
    return display;
  });

  renderer.setSetting("edgeReducer", (edge, attrs) => {
    const display = { ...attrs };
    if (selection.hoveredId !== null) {
      const hoveredKey = String(selection.hoveredId);
      const extremities = graph.extremities(edge);
      if (!extremities.includes(hoveredKey)) {
        display.hidden = true;
      }
    }
    return display;
  });

  const searchInput = document.getElementById("search") as HTMLInputElement;
  const resultsEl = document.getElementById("search-results") as HTMLDivElement;

  let debounceHandle: ReturnType<typeof setTimeout> | undefined;
  searchInput.addEventListener("input", () => {
    clearTimeout(debounceHandle);
    debounceHandle = setTimeout(() => {
      const matches = searchNames(data.nodes, searchInput.value);
      resultsEl.innerHTML = "";
      for (const node of matches) {
        const item = document.createElement("div");
        item.className = "search-result";
        item.textContent = node.name;
        item.addEventListener("click", () => {
          selection.selectedId = node.id;
          flyToNode(renderer, graph, String(node.id));
          resultsEl.innerHTML = "";
          searchInput.value = node.name;
          renderSidebar();
        });
        resultsEl.appendChild(item);
      }
    }, 100);
  });
}

bootstrap().catch((err) => {
  console.error(err);
});
```

- [ ] **Step 6: Manually verify in the browser**

```bash
npm run dev
```

Confirm: hovering a node dims non-neighbors and hides non-connecting edges; clicking a node opens the sidebar with thumbnail, name, attribution, and ranked similar list with percentages; clicking a similar-list entry navigates to that node; pressing Esc or clicking empty canvas closes the sidebar.

- [ ] **Step 7: Commit**

```bash
cd /c/Users/danie/projects/doppelmap
git add code/web/src/interactions.ts code/web/src/main.ts code/web/tests/interactions.test.ts
git commit -m "feat: add hover highlighting, click-to-inspect sidebar, and deselect"
```

---

## Task 11: Mobile responsiveness, README, end-to-end verification

**Files:**
- Modify: `code/web/src/style.css` (already has the `@media (max-width: 640px)` bottom-sheet rule from Task 7 — verify it, extend if needed)
- Create: `code/README.md`

**Interfaces:**
- Consumes: everything from Tasks 1–10.
- Produces: nothing new consumed elsewhere — this is the final polish/documentation task.

- [ ] **Step 1: Verify mobile layout manually**

```bash
cd code/web
npm run dev
```

Open devtools, toggle device toolbar to a phone width (e.g. 390px), reload. Confirm the sidebar becomes a bottom sheet (fixed to the bottom, ~40vh tall, full width) instead of the right-hand panel, and the search bar remains usable at the top. If the sidebar doesn't reflow correctly, adjust the `@media (max-width: 640px)` block in `code/web/src/style.css` (from Task 7) until it does.

- [ ] **Step 2: Write the README**

`code/README.md`:
```markdown
# Doppelmap — local dev

Two pieces: a Python pipeline that generates a synthetic celebrity
similarity graph, and a Vite/TypeScript frontend that renders it. See
`../doppelmap-lld.md` for the full design; this covers just how to run
what's in this plan.

## 1. Generate the dataset

```bash
cd similarity_map
python -m pip install -r requirements.txt
python -m pytest -v          # run the pipeline test suite
python -m pipeline.build_dataset --count 150 --k 8 --seed 42
```

This writes `../web/public/data/graph.json` and
`../web/public/data/thumbs/*.webp`. All embeddings, similarities, and
thumbnails are **synthetic placeholders** — see `doppelmap-lld.md` for the
real (Wikidata + ArcFace) pipeline this will eventually be replaced with.

## 2. Run the frontend

```bash
cd web
npm install
npm test          # run the frontend test suite (Vitest)
npm run dev        # start the dev server and open the printed URL
```

## Regenerating with a different size

```bash
python -m pipeline.build_dataset --count 80 --k 6 --seed 7
```

Re-run `npm run dev` (or just refresh the page) afterward — no rebuild step
is needed since `graph.json` is fetched at runtime from `public/data/`.
```

- [ ] **Step 3: Full end-to-end verification checklist**

Run through this manually with the dev server running (`npm run dev` in `code/web`), dataset already generated:

- [ ] Page loads with ~150 nodes visible, laid out in visible cluster structure (not a uniform blob).
- [ ] Zooming in reveals name labels; zooming out hides them.
- [ ] Typing a known name (e.g. "Tom") in search shows a dropdown with matches, prefix matches first.
- [ ] Clicking a search result flies the camera to that node and opens the sidebar.
- [ ] Hovering any node dims unrelated nodes/edges and highlights its mutual neighbors.
- [ ] Clicking a node opens the sidebar: placeholder thumbnail, name, "Synthetic placeholder — no real photo", and a ranked list of similar people with percentages.
- [ ] Clicking an entry in that ranked list navigates to the new node and updates the sidebar.
- [ ] Pressing Esc, and separately clicking empty canvas, both close the sidebar.
- [ ] Resizing to a mobile width turns the sidebar into a bottom sheet.

- [ ] **Step 4: Run both full test suites one last time**

```bash
cd similarity_map && python -m pytest -v
cd ../web && npm test
```

Expected: all tests passing in both suites.

- [ ] **Step 5: Commit**

```bash
cd /c/Users/danie/projects/doppelmap
git add code/web/src/style.css code/README.md
git commit -m "docs: add run instructions and verify mobile layout"
```

---

## Self-Review Notes

- **Spec coverage:** every `graph.json` field (LLD §2.5), the mutual-kNN + directed-similar split (§2.3), UMAP + `[0,10000]²` normalization (§2.4), and the full V1 interaction table (§3.2: pan/zoom, hover, click, similar-list click, search select, Esc/background deselect) each map to a task above. Explicitly deferred: real image sourcing (§2.1), RetinaFace/ArcFace (§2.2), hosting/CDN (§4), and WebGL thumbnail sprites — all called out in Global Constraints as out of scope for this synthetic-data plan.
- **Type consistency:** `GraphNode`/`GraphData` (Task 7) are reused unchanged by `search.ts` (Task 9), `interactions.ts` (Tasks 9–10), and `main.ts` throughout; Python's `build_dataset` node dict keys (`id, name, x, y, deg, thumb, attr`) match `GraphNode` exactly.
- **No placeholders:** all code blocks are complete and runnable as written.
