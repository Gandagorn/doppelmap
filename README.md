# Doppelmap

**[gandagorn.github.io/doppelmap](https://gandagorn.github.io/doppelmap/)**

An interactive map of face similarity between public figures. Every point is a
real person; people who look alike end up near each other, connected by a line.
1,701 people, 2,979 connections.

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
- **Share** — a comparison shares as a picture of itself: the two faces, the
  score and the verdict, 1200x630 so it survives any link preview.
- **Page through comparisons** — Prev/Next, or the arrow keys, walk whichever
  ranking you opened: the Top Pairs table, or one person's own matches.
- **Vote on a pair** — say whether two people actually look alike, and see
  what everyone else said. Votes are stored against Wikidata ids in Supabase,
  one per person per pair, changeable. Sign-in is anonymous and silent, which
  means the totals are a mood rather than a poll.

## How it works

The map is built from a dataset prepared offline and committed here as two
files: `graph.json` — who is on the map, where they sit and how strongly they
match — and `photos.json`, which Commons photograph to show for each person
and where the face sits inside it.

1. **People.** Drawn from the most-viewed English Wikipedia biographies,
   restricted to real people. Some are deliberately left out; see below.
2. **Photographs.** Wikimedia Commons, CC-licensed or public domain, credited
   on every picture shown. Nothing is re-hosted: the site links Commons and
   cuts the face out in the browser.
3. **Comparable similarity.** Faces are compared with a face-recognition
   model, with the population's average face subtracted first — face vectors
   share a large generic-face component, and removing it means a score
   reflects what makes two people look alike rather than what all faces have
   in common.
4. **Similarity graph.** Two people are connected only when each is among the
   other's nearest matches (mutual k-nearest-neighbours, k=5), which keeps the
   graph sparse and readable instead of all-to-all.
5. **Layout.** [UMAP](https://umap-learn.readthedocs.io/) projects the
   comparison space to two dimensions, with a soft compression of the outer
   tail so a handful of outliers cannot squash everyone else into the middle.
6. **Rendering.** [Sigma.js](https://www.sigmajs.org/) draws the graph in
   WebGL. The face on each node is cut from the Commons photograph in the
   browser, on demand, for whoever is on screen.

The site is static: no backend, no database, and no images of its own. A
GitHub Actions workflow deploys to GitHub Pages on every push to `main`.

## What is left out

Perpetrators of mass or violent crime are not on the map: a lookalike result
would pair a living person with them by name and face. Neither are people
known chiefly as the victim of a violent crime, whose photographs should not
be ranked for resemblance. Heads of state and politicians stay, however they
are thought of, and so do people who have only been accused of something —
leaving those out would read as a verdict.

## What it is not

Face recognition is a narrower thing than "these two look alike" as a person
would mean it.

It measures bone structure, so it rates **relatives** very highly — siblings
and parents reliably outrank the lookalike pairs people actually name.

It was also never trained on **artwork**, so painted portraits, engravings and
busts resemble each other partly because of the medium rather than the face:
two 16th-century portraits once outranked every genuine lookalike on the map.
Artwork is detected from Wikimedia Commons categories and creation dates and
excluded. Anyone who died before 1845 is removed outright, from their Wikidata
death date: per-image detection is not enough on its own, since Jane Austen's
engravings are dated 1870 — the year they were printed — and name no medium,
while the files that survive such a cull turn out to be different people who
happen to share the name.

Coverage depends on someone having usable freely-licensed photographs on
Commons. Plenty of well-known people have neither.

## Repository layout

```
code/web/              Vite + TypeScript frontend (Sigma.js / graphology)
code/web/public/data/  The built map: graph.json and photos.json
.github/workflows/     GitHub Pages deploy
```

See [`code/README.md`](code/README.md) for running it locally.

## Licence

The code is [MIT](LICENSE).

The photographs are not covered by it. Each one keeps the licence it carries
on Wikimedia Commons, shown with the picture.
