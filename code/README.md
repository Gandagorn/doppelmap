# Doppelmap — local dev

Two pieces: a Python pipeline that assembles a celebrity similarity graph,
and a Vite/TypeScript frontend that renders it. See `../doppelmap-lld.md`
for the full design; this covers just how to run what's here.

## 1. Generate the dataset

The pipeline has two modes:

**Real embeddings** (default source now) — load ArcFace embeddings from a
`.npz` produced by the notebook pipeline described in the design doc
(`names` + `E` arrays, one 512-d L2-normalized vector per name):

```bash
cd similarity_map
python -m pip install -r requirements.txt
python -m pytest -v          # run the pipeline test suite
python -m pipeline.build_dataset --embeddings ../../data/prototypes.npz --k 8 --seed 42
```

Every name in the file is used — there's no count cap. Profile photos are
**not** fetched here; the frontend looks them up live from Wikipedia when a
node is clicked, so this stays fully offline regardless of dataset size.

**Synthetic embeddings** (for prototyping without real data):

```bash
python -m pipeline.build_dataset --count 150 --k 8 --seed 42
```

Both modes write `../web/public/data/graph.json` and
`../web/public/data/thumbs/*.webp`.

## 2. Run the frontend

```bash
cd web
npm install
npm test          # run the frontend test suite (Vitest)
npm run dev        # start the dev server and open the printed URL
```

Re-run `npm run dev` (or just refresh the page) after regenerating the
dataset — no rebuild step is needed since `graph.json` is fetched at
runtime from `public/data/`.
