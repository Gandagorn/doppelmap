# Doppelmap — Low-Level Design (V1)

**Working name:** Doppelmap (alternates: Doppelgraph, FaceAtlas, Lookalike.space)

**V1 scope:** A single-page website rendering a precomputed similarity graph of celebrity faces. The user can pan, zoom, search for a person, and click a node to see their nearest lookalikes. Nothing else: no user uploads, no accounts, no server-side compute at request time.

---

## 1. Architecture overview

The system is split into an **offline pipeline** (Python, run once per dataset release) and a **static frontend** (browser-only). V1 has no backend service at all — the pipeline emits static JSON + thumbnails, and the site is served from a CDN. This is the cheapest and most robust possible deployment, and nothing in the V1 feature list (render, search, navigate) requires a server.

```
[Image sources] → [Pipeline: detect → embed → aggregate → kNN → layout] → graph.json + thumbs/
                                                                              │
                                                    [Static hosting / CDN] ←──┘
                                                              │
                                                        [Browser SPA]
```

## 2. Offline pipeline

### 2.1 Data sourcing

Use **Wikimedia Commons portraits resolved via Wikidata** rather than scraped press photos. Wikidata entities for people carry an `image (P18)` claim pointing at a freely licensed Commons file, which sidesteps the licensing problem that killed MS-Celeb-1M-style datasets. Seed list: Wikidata SPARQL query for humans with occupation ∈ {actor, musician, athlete, politician, …} ranked by sitelink count, capped at N = 2,000 for V1.

Per person, fetch 1 canonical image (the P18 image). Multi-image averaging improves embedding quality, but Commons rarely has 5–10 clean portraits per person; V1 accepts single-image embeddings and treats multi-image aggregation as a quality upgrade later. Store attribution metadata (author, license) per image — Commons licenses (CC BY, CC BY-SA) require it.

### 2.2 Face processing

1. **Detection + alignment:** RetinaFace (via `insightface`). Reject images where zero or >1 confident faces are found, or where the face box is < 80 px. Expect ~10–15% attrition; over-seed the candidate list accordingly.
2. **Embedding:** ArcFace (`insightface` `buffalo_l`, 512-d), L2-normalized. ArcFace is trained to separate identities, which works against "perceptual lookalike" similarity at the margins, but with one canonical vector per person its nearest-neighbor structure still tracks perceived resemblance well, and it is far more robust to pose/lighting than CLIP.
3. **Thumbnail:** aligned face crop, 96×96, WebP quality 80 (~3–4 KB each). ~2,000 thumbs ≈ 7 MB total, lazy-loaded.

### 2.3 Graph construction

- Cosine kNN with k = 8 over the normalized embeddings (`sklearn.neighbors` brute force is fine at N = 2k; switch to FAISS/hnswlib only if N grows past ~50k).
- **Mutual-kNN filter:** keep edge (a, b) only if a ∈ kNN(b) and b ∈ kNN(a). This kills hub nodes (generic-looking faces that appear in everyone's neighbor list) and makes clusters visually crisper. Keep the raw directed top-8 list per node separately for the sidebar ("most similar to X"), because the mutual filter is too aggressive for a ranked list.
- Edge weight = cosine similarity, stored as float rounded to 3 decimals.
- Drop isolated nodes or attach each to its single nearest neighbor with a flagged "weak" edge — decide by looking at how many isolates the mutual filter produces (expected: a few dozen).

### 2.4 Layout

Precompute node positions in the pipeline; do **not** run force layout in the browser (non-deterministic, slow on load, and positions should be stable across sessions for shareable URLs later).

- Initialize positions with UMAP (2-d, cosine metric, `n_neighbors=15`) for global structure, then refine with ForceAtlas2 (`fa2`) on the mutual-kNN graph for ~500 iterations so edge lengths reflect graph distance rather than UMAP's distorted metric.
- Normalize coordinates to a fixed canvas space, e.g. [0, 10000]².
- Important honesty note carried into the UI: **2-D proximity is suggestive, edges are authoritative.** The frontend should emphasize edges and the ranked-similar list, not raw map distance.

### 2.5 Output artifacts

```jsonc
// graph.json  (~1.5–2 MB raw, ~400 KB gzipped at N=2k)
{
  "meta": { "version": "2026-08-29", "count": 2000, "k": 8 },
  "nodes": [
    {
      "id": 17,                    // dense int index
      "name": "…",
      "x": 4211.3, "y": 8022.7,
      "deg": 6,                    // mutual-graph degree, drives node size
      "thumb": "t/17.webp",
      "attr": "Photo: J. Doe, CC BY-SA 4.0"  // Commons attribution
    }
  ],
  "edges": [[17, 342, 0.612], ...],          // [src, dst, weight]
  "similar": { "17": [[342, 0.612], ...] }   // top-8 directed, for sidebar
}
```

Plus `thumbs/` directory and a tiny `search-index.json` (id → lowercase name) if `graph.json` isn't loaded eagerly — in V1 it is, so search runs over the in-memory node list and no separate index file is needed.

## 3. Frontend

### 3.1 Stack

- **Rendering: Sigma.js v3 + graphology.** WebGL renderer, comfortably handles 2k nodes / ~10k edges with image nodes, and its API is built around exactly this use case (hover neighbor highlighting, camera animation to a node). Alternatives considered: Cosmograph (great but GPU-layout-oriented; we have precomputed layout), deck.gl (more code for the same result), plain D3/canvas (too slow with image sprites at this scale).
- **App shell:** Vite + vanilla TypeScript or Preact. No router, no state library — total UI state is `{selectedNode, hoveredNode, searchQuery, camera}`.
- **Search:** client-side substring/prefix match over node names (2k strings — no library needed; normalize diacritics with `String.prototype.normalize('NFD')`). Debounced 100 ms, dropdown of top 8 matches with thumbnails.

### 3.2 Interactions (V1 complete list)

| Interaction | Behavior |
|---|---|
| Pan / zoom | Sigma camera; pinch + wheel; zoom-dependent label display (labels only above a zoom threshold, thumbnails swap in for dots above another threshold) |
| Hover node | Highlight node + mutual-graph neighbors + connecting edges; dim the rest |
| Click node | Animate camera to node, open sidebar: large thumb, name, attribution, ranked top-8 similar list (from `similar`, with similarity shown as a percentage) |
| Click similar-list entry | Select that node (camera flies over — this is the primary "navigate the graph" loop) |
| Search select | Same as click node |
| Esc / click background | Deselect, close sidebar |

### 3.3 Loading & performance budget

- First paint: shell + spinner < 50 KB. `graph.json` gzipped ~400 KB fetched immediately; render dots+edges as soon as parsed (< 1 s on decent connections).
- Thumbnails lazy: only fetch thumbs for nodes within the current viewport above the thumbnail zoom threshold, plus the selected node's sidebar list. Simple LRU cap of ~500 decoded images.
- Target: 60 fps pan/zoom on a mid-range phone; Sigma with 2k nodes is well within this.

### 3.4 Mobile

Same SPA, responsive: sidebar becomes a bottom sheet; search bar fixed top. Touch targets = node hit area inflated ~1.5× at low zoom.

## 4. Hosting & ops

- Static hosting on Cloudflare Pages or GitHub Pages; thumbs and graph.json served with `Cache-Control: immutable` and a version hash in the path (`/v/2026-08-29/graph.json`), so a dataset release is just a new directory + updated pointer.
- Pipeline runs locally or in a GitHub Action (CPU-only is fine: 2k images through RetinaFace+ArcFace ≈ minutes).
- No analytics/backend in V1; optional privacy-friendly counter (e.g., Cloudflare Web Analytics) later.

## 5. Risks & mitigations

- **Similarity quality disappoints** (ArcFace separates identities too well): mitigation A — blend ArcFace with a CLIP image embedding (weighted concat, e.g. 0.7/0.3) and eyeball neighbor lists; mitigation B — add per-person multi-image averaging. Decide from a manual review of ~50 random nodes' top-8 lists before shipping.
- **Commons image quality is uneven** (old photos, odd angles): detection-stage quality gates (face size, blur via Laplacian variance) plus manual blacklist file consumed by the pipeline.
- **Legal:** stick to freely licensed images with attribution; the site is informational/entertainment, no endorsement implied. Add a short about/disclaimer page.

## 6. Out of scope for V1 (V2 candidates)

Upload-your-face lookup (requires server or on-device ONNX inference), shareable deep links to a selected node (trivial: `#id` hash — likely the first fast-follow), clustering labels/regions, filters by occupation or era, multi-image embeddings, larger N with tiled/level-of-detail loading.
