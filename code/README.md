# Doppelmap — local dev

Two pieces: a Python pipeline that assembles the celebrity similarity graph
(one file per popularity level), and a Vite/TypeScript frontend that renders
it.

## 1. Generate the dataset

Loads ArcFace embeddings from a `.npz` produced by the notebook pipeline
(`data/doppelmap_ipynb.py`; `names` + `E` + `n_used` arrays, one 512-d
L2-normalized vector per name):

```bash
cd similarity_map
python -m pip install -r requirements.txt
python -m pytest -v          # run the pipeline test suite
python -m pipeline.build_dataset --embeddings ../../data/prototypes.npz --k 8 --seed 42
```

Every name in the file is used — there's no count cap. Profile photos are
**not** fetched here; the frontend looks them up live from Wikipedia when a
node is clicked, so this stays fully offline regardless of dataset size.

Writes four independent datasets to `../web/public/data/` — `graph-all.json`,
`graph-top50.json`, `graph-top20.json`, `graph-top5.json` (each with its own
kNN graph and layout, computed only among that level's members) — plus
`thumbs/*.webp`, shared by content-hash filename across levels.

## 2. Run the frontend

```bash
cd web
npm install
npm test          # run the frontend test suite (Vitest)
npm run dev        # start the dev server and open the printed URL
```

Re-run `npm run dev` (or just refresh the page) after regenerating the
dataset — no rebuild step is needed since the graph files are fetched at
runtime from `public/data/`.
