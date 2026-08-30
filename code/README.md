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
