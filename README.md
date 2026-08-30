# Doppelmap

**[gandagorn.github.io/doppelmap](https://gandagorn.github.io/doppelmap/)**

An interactive map of celebrity face-similarity, built from real face embeddings. Every point is a real public figure; people who look alike end up near each other on the map, connected by a line.

## What it does

- **Explore the map** — pan and zoom a WebGL-rendered graph of thousands of people, positioned so visually-similar faces cluster together.
- **Search** by name, or click any node to open a sidebar with their photo, a Wikipedia link, and their most similar matches.
- **Walk the Graph** — auto-advance from person to person along the strongest similarity match, like a guided tour of lookalikes.
- **Fame level slider** — narrow the map down to the top 5/20/50% most-photographed people, or show everyone.
- **Top Pairs** — a running list of the single most similar pairs on the current map.
- Every fame level, and whoever's selected, is reflected in the URL — reload or share the link and you're back where you left off.

## How it works, briefly

1. **Embeddings**: real photos (from the [IMDB-WIKI](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/) dataset) are run through [InsightFace's ArcFace](https://github.com/deepinsight/insightface) model, turning each photo into a 512-dimensional vector where similar-looking faces land close together.
2. **One point per person**: each person's photos are reduced to a single consensus embedding — a medoid-and-filter step that throws out mislabeled or low-quality crops before averaging.
3. **Similarity graph**: people are connected by an edge only when each is genuinely among the other's closest matches (mutual k-nearest-neighbors), keeping the graph sparse and readable instead of an all-to-all mess.
4. **Layout**: [UMAP](https://umap-learn.readthedocs.io/) projects the 512 dimensions down to 2D, then a custom pass nudges connected people closer together so edge length on the map actually reflects how similar two people are.
5. **Rendering**: [Sigma.js](https://www.sigmajs.org/) (WebGL) draws the graph in the browser; profile photos are looked up live from Wikipedia the moment you click someone, so the dataset itself stays lightweight.

The whole site is static — no backend, no database. A GitHub Actions workflow rebuilds and redeploys it to GitHub Pages on every push to `main`.

## Repository layout

```
code/similarity_map/   Python pipeline: embeddings -> similarity graph -> 2D layout
code/web/              Vite + TypeScript frontend (Sigma.js / graphology)
data/                  Embedding data and the notebook that generates it
.github/workflows/     GitHub Pages deploy
```

See [`code/README.md`](code/README.md) for how to regenerate the dataset and run the frontend locally.

## Coverage

The map is only as complete as its source data: [IMDB-WIKI](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/) is a snapshot of celebrity photos crawled from IMDb and Wikipedia around 2015. Anyone who wasn't already well-photographed by then, or who rose to fame since, won't appear.
