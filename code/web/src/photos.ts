import type { PhotoData, PhotoRef } from "./types";

/** Commons serves a resized copy through Special:FilePath, picking a valid size itself. */
export function photoUrl(photo: PhotoRef, width = 640): string {
  // An uploaded photo is already a blob: URL in this tab.
  if (photo.f.startsWith("blob:") || photo.f.startsWith("data:")) return photo.f;
  // encodeURIComponent leaves !
  const name = encodeURIComponent(photo.f.replace(/ /g, "_")).replace(
    /[!'()*]/g,
    (c) => `%${c.charCodeAt(0).toString(16).toUpperCase()}`
  );
  return `https://commons.wikimedia.org/wiki/Special:FilePath/${name}?width=${width}`;
}

/** CSS that shows just the face inside a fixed-size box. */
export function faceCropStyle(photo: PhotoRef, padding = 0.55, displayPx = 160): string {
  const [x1, y1, x2, y2] = photo.b;
  const w = Math.max(x2 - x1, 0.001);
  const h = Math.max(y2 - y1, 0.001);
  const cx = x1 + w / 2;
  const cy = y1 + h / 2;

  // Zoom so the whole padded face fits, choosing whichever axis binds.
  const aspect = photo.a > 0 ? photo.a : 1;
  const fitWidth = 1 / (w * (1 + padding));
  const fitHeight = aspect / (h * (1 + padding));
  const zoom = Math.max(Math.min(fitWidth, fitHeight), 1, aspect);

  // background-position aligns the P% point of the image with the P% point of the box,
  // so centring a point needs this rather than a raw offset.
  const place = (c: number, extent: number) => {
    const span = 1 - 1 / (zoom * extent);
    if (!(span > 0)) return 50;
    return Math.min(100, Math.max(0, ((c - 1 / (2 * zoom * extent)) / span) * 100));
  };

  // Ask Commons for a source big enough that the zoomed face still has pixels.
  const needed = Math.min(2400, Math.max(320, Math.round((displayPx * zoom) / 160) * 160));
  return [
    // Single quotes: this lands in a style="..." attribute, and a double quote inside it
    // would close the attribute and drop the rest of the rule on the floor.
    `background-image:url('${photoUrl(photo, needed)}')`,
    `background-size:${(zoom * 100).toFixed(1)}% auto`,
    `background-position:${place(cx, 1).toFixed(1)}% ${place(cy, 1 / aspect).toFixed(1)}%`,
    "background-repeat:no-repeat",
  ].join(";") + ";";
}

let cache: PhotoData | null = null;
let pending: Promise<PhotoData> | null = null;

/** photos.json is fetched on first use, not at startup. */
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

/** Photos for people who exist only in this tab, keyed by their negative id. */
const uploaded = new Map<number, PhotoRef[]>();

export function rememberUploadedPhotos(id: number, photos: PhotoRef[]): void {
  uploaded.set(id, photos);
}

/** Photos already in memory, or undefined if the file has yet to arrive. */
export function photosFor(id: number): PhotoRef[] | undefined {
  return uploaded.get(id) ?? cache?.[String(id)];
}
