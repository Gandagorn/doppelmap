import type { PhotoData, PhotoRef } from "./types";

/** Commons serves a resized copy through Special:FilePath, picking a valid
 *  size itself. Deriving an upload.wikimedia.org /thumb/ URL directly does
 *  not work: the permitted widths are a per-image bucket list and anything
 *  else is refused. */
export function photoUrl(photo: PhotoRef, width = 640): string {
  const name = encodeURIComponent(photo.f.replace(/ /g, "_"));
  return `https://commons.wikimedia.org/wiki/Special:FilePath/${name}?width=${width}`;
}

/** CSS that shows just the face inside a fixed-size box.
 *
 *  Emits background-image rules rather than sizing an <img>, because a
 *  background is clipped by its own element no matter how far it is zoomed.
 *  The first attempt scaled an <img> with a transform, which the parent's
 *  overflow could not contain: enlarged thumbnails spilled across the names
 *  beside them.
 *
 *  The face box is stored as fractions of the image, so the crop works
 *  against whatever size Commons serves, and the image's aspect ratio is
 *  stored alongside it. That ratio cannot be inferred from the box:
 *  detector boxes run about 1:1.85 tall, so treating a face as square put
 *  the crop on empty background.
 *
 *  Padding keeps some hair and chin in frame -- cropped tight to the
 *  detector's bounds, a portrait reads as a mugshot. */
export function faceCropStyle(photo: PhotoRef, padding = 0.55, displayPx = 160): string {
  const [x1, y1, x2, y2] = photo.b;
  const w = Math.max(x2 - x1, 0.001);
  const h = Math.max(y2 - y1, 0.001);
  const cx = x1 + w / 2;
  const cy = y1 + h / 2;

  // Zoom so the padded face box fills the frame. Never below 1: shrinking
  // would letterbox the image inside its own element.
  const zoom = Math.max(1 / (w * (1 + padding)), 1);

  // background-position aligns the P% point of the image with the P% point
  // of the box, so centring a point needs this rather than a raw offset.
  const place = (c: number, extent: number) => {
    const span = 1 - 1 / (zoom * extent);
    if (!(span > 0)) return 50;
    return Math.min(100, Math.max(0, ((c - 1 / (2 * zoom * extent)) / span) * 100));
  };

  // Width drives the scale. `background-size: Z% auto` renders the height
  // as Z/aspect of the box, so vertical magnification is the zoom divided
  // by the image's aspect ratio.
  const aspect = photo.a > 0 ? photo.a : 1;

  // Ask Commons for an image large enough that the face still has pixels
  // after the zoom. A face filling 5% of the frame shown at 160px needs a
  // ~3000px source; requesting a flat 640px would render about thirty
  // pixels of face, stretched, which is how several portraits came out as
  // a smear of backdrop. Capped so a small thumbnail never pulls a huge
  // file, and rounded to keep Commons caching the same few sizes.
  const needed = Math.min(2400, Math.max(320, Math.round((displayPx * zoom) / 160) * 160));
  return [
    // Single quotes: this lands in a style="..." attribute, and a double
    // quote inside it would close the attribute and drop the rest of the
    // rule on the floor.
    `background-image:url('${photoUrl(photo, needed)}')`,
    `background-size:${(zoom * 100).toFixed(1)}% auto`,
    `background-position:${place(cx, 1).toFixed(1)}% ${place(cy, 1 / aspect).toFixed(1)}%`,
    "background-repeat:no-repeat",
  ].join(";") + ";";
}

let cache: PhotoData | null = null;
let pending: Promise<PhotoData> | null = null;

/** photos.json is fetched on first use, not at startup.
 *
 *  It is several times the size of the graph and only matters once someone
 *  opens a person, so making every visitor wait for it would be a poor
 *  trade. */
export function loadPhotos(url: string): Promise<PhotoData> {
  if (cache) return Promise.resolve(cache);
  if (!pending) {
    pending = fetch(url)
      .then((r) => (r.ok ? r.json() : {}))
      .then((data: PhotoData) => {
        cache = data;
        return data;
      })
      .catch(() => ({}) as PhotoData);
  }
  return pending;
}

/** Photos already in memory, or undefined if the file has yet to arrive. */
export function photosFor(id: number): PhotoRef[] | undefined {
  return cache?.[String(id)];
}
