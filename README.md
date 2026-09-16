# Doppelmap

**[gandagorn.github.io/doppelmap](https://gandagorn.github.io/doppelmap/)**

An interactive map of face similarity between public figures, built from real
face embeddings. Every point is a real person; people who look alike end up
near each other, connected by a line. 1,721 people, 3,675 connections.

## What it does

- **Explore the map** — pan and zoom a WebGL graph positioned so visually
  similar faces cluster together.
- **Search** by name, or click anyone to open their photo, a Wikipedia link,
  a one-line description, and their closest matches.
- **Compare two faces** — the ⇄ button shows the two photographs that
  actually match. Not two arbitrary portraits: the specific pair the model
  rates highest, worked out when the dataset is built.
- **Plain-language scores** — matches are grouped as *Lookalike*, *Looks
  somewhat alike* and *Far resemblance* rather than shown as bare
  percentages, because the raw numbers sit in a narrow band where the
  difference between 15% and 17% is noise.
- **Walk the Graph** — hop from person to person along the strongest match,
  as a guided tour of lookalikes.
- **Top Pairs** — the most similar pairs anywhere on the map.
- **Share** — a link to a person, or to a specific comparison, that reopens
  exactly what you were looking at.

## How it works

1. **Collection.** For each person, Wikimedia Commons is searched for
   photographs and every face in each image is detected and embedded with
   [InsightFace](https://github.com/deepinsight/insightface)'s ArcFace
   (`buffalo_l`), averaged with its mirror image. Faces, licences and
   credits are kept per image, along with the pixel size the face box was
   measured in — Commons serves a bucketed thumbnail size, not the one you
   ask for, and without recording it the boxes cannot be scaled correctly
   later.
2. **One point per person.** A gallery is rarely clean: a search for someone
   returns group shots, other people entirely, and the occasional wrong
   subject. Images whose *title* names the person act as an identity anchor,
   deciding which face in the pile is actually them; the rest are accepted
   or rejected by how close they sit to that consensus. Nobody is dropped
   for a missing title match alone.
3. **Comparable similarity.** Every prototype has the population's average
   face subtracted before anything is compared. ArcFace vectors share a
   large generic-face component, and removing it means a score reflects what
   makes two people look alike rather than what all faces have in common.
   Galleries that turn out to be the same person under two spellings — or a
   search that quietly returned somebody else — are detected and removed.
4. **Similarity graph.** Two people are connected only when each is among
   the other's nearest matches (mutual k-nearest-neighbours, k=6), which
   keeps the graph sparse and readable instead of all-to-all.
5. **Layout.** [UMAP](https://umap-learn.readthedocs.io/) projects 512
   dimensions to 2, with a soft compression of the outer tail so a handful
   of outliers cannot squash everyone else into the middle.
6. **Rendering.** [Sigma.js](https://www.sigmajs.org/) draws the graph in
   WebGL. Photographs are fetched from Commons at display time and cropped
   to the stored face box in CSS, so the dataset stays small — 743 KB of
   graph and 1.4 MB of photo references for 8,813 faces.

The whole site is static: no backend, no database. A GitHub Actions workflow
rebuilds and redeploys to GitHub Pages on every push to `main`.

## Repository layout

```
code/similarity_map/   Python pipeline: embeddings -> similarity graph -> 2D layout
code/web/              Vite + TypeScript frontend (Sigma.js / graphology)
data/                  Collector notebook and the per-person face data it writes
.github/workflows/     GitHub Pages deploy
```

See [`code/README.md`](code/README.md) for how to regenerate the dataset and
run the frontend locally.

## What it is not

Similarity here is cosine distance between ArcFace embeddings, and that is a
narrower thing than "these two look alike" as a person would mean it.

It measures bone structure, so it rates **relatives** very highly — siblings
and parents reliably outrank the lookalike pairs people actually name. It was
also never trained on **artwork**, so painted portraits, engravings and busts
resemble each other partly because of the medium; historical figures cluster
for that reason as much as any other.

Coverage comes from the most-viewed English Wikipedia biographies, restricted
to real people, and depends on someone having usable freely-licensed
photographs on Commons. Plenty of well-known people have neither.

All images are CC-licensed or public domain and credited on the photo itself.

## Licence

The code is [MIT](LICENSE).

The photographs are not covered by it. Each one keeps the licence it carries
on Wikimedia Commons — CC BY-SA, CC0 or public domain — and is credited to
its photographer in the sidebar and on the comparison card. Nothing is
rehosted: images are requested from Commons at display time, and the
repository stores only a filename, a face box and the credit line.
