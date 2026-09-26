# Running the site

A Vite/TypeScript frontend that renders a prepared dataset. Both files it
needs are committed under `web/public/data/`, so there is nothing to generate
first.

```bash
cd web
npm install
npm run dev        # the printed URL
npm test           # Vitest
npm run typecheck
```

## The data it reads

`graph.json` — who is on the map, where each person sits, how strongly they
match, and the pairs the landing view opens with.

`photos.json` — per person, best photograph first: the Commons filename, the
face box as fractions of the image, the image's aspect ratio, and the credit
and licence to show. The first entry also carries `u`, a thumbnail URL on
Wikimedia's image host.

`u` is there because the map has a constraint the other surfaces do not.
Sigma uploads node images into a WebGL texture, so the image must arrive with
CORS headers; `Special:FilePath` redirects through a host that sends none.
`upload.wikimedia.org` does send them but serves only certain widths per
image — 330px of a given photograph is HTTP 200 while 320px is HTTP 400 — so
the URL is resolved when the dataset is built rather than guessed at here.

## Faces on the map

`faceTiles.ts` fetches each thumbnail, crops it to the stored face box on a
small canvas, and hands the result to Sigma as a blob URL, nearest the centre
of the screen first. Cut tiles are kept in the Cache Storage API, so panning
back and returning later need no traffic.

The face comes from `photos.json`, which is written by the same build as
`graph.json`: one source of truth, so a face cannot end up on the wrong
person.

Refreshing the page is enough after the data changes — the JSON is fetched at
runtime.
