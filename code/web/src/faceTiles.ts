/** Faces on the map, cut from the Commons photograph in the browser.
 *
 *  Sigma uploads node images into a WebGL texture, so the image has to arrive
 *  with CORS headers, and a photograph is rarely centred on the face. Each one
 *  is fetched from the thumbnail host, cropped to its stored box and handed
 *  back as a blob URL. A pool of workers does the cutting, because the canvas
 *  work rather than the network is what limits how many can be prepared at
 *  once.
 */
import type { PhotoRef } from "./types";
import type { CutRequest, CutResult } from "./faceWorker";

const TILE = 72;
/** Matches faceCropStyle in photos.ts, so map and sidebar frame a face alike. */
const PADDING = 0.55;
const WORKERS = 6;   // 24 fetches in flight: more, and Wikimedia starts refusing
const PER_WORKER = 4;

const done = new Map<string, string>();
const failed = new Set<string>();
const queued = new Set<string>();
const queue: { id: string; photo: PhotoRef }[] = [];
let onReady: (id: string, url: string) => void = () => {};
let onIdle: (() => void) | undefined;
let cutSinceIdle = 0;
let pool: { worker: Worker; busy: number }[] | undefined;

/** The crop, in fractions of the image: the face box plus a little room. */
export function cropBox(photo: PhotoRef): { x: number; y: number; size: number } {
  const [x1, y1, x2, y2] = photo.b;
  const w = Math.max(x2 - x1, 0.001);
  const h = Math.max(y2 - y1, 0.001);
  const aspect = photo.a > 0 ? photo.a : 1;
  // A square in pixels is not a square in fractions: the image is `aspect`
  // times wider than tall, so the same pixel width is a smaller fraction
  // across than down.
  const side = Math.max(w * (1 + PADDING), (h * (1 + PADDING)) / aspect);
  return {
    x: Math.min(Math.max(x1 + w / 2 - side / 2, 0), Math.max(0, 1 - side)),
    y: Math.min(Math.max(y1 + h / 2 - (side * aspect) / 2, 0),
                Math.max(0, 1 - side * aspect)),
    size: Math.min(side, 1),
  };
}

function workers(): { worker: Worker; busy: number }[] {
  if (pool) return pool;
  pool = Array.from({ length: WORKERS }, () => {
    const worker = new Worker(new URL("./faceWorker.ts", import.meta.url), { type: "module" });
    const slot = { worker, busy: 0 };
    worker.onmessage = (event: MessageEvent<CutResult>) => {
      const { id, blob, error } = event.data;
      slot.busy -= 1;
      if (blob && !error) {
        const url = URL.createObjectURL(blob);
        done.set(id, url);
        onReady(id, url);
      } else {
        failed.add(id);
      }
      pump();
      if (!queue.length && pool?.every((s) => !s.busy) && cutSinceIdle) {
        cutSinceIdle = 0;
        onIdle?.();
      }
    };
    return slot;
  });
  return pool;
}

function pump(): void {
  for (const slot of workers()) {
    while (slot.busy < PER_WORKER && queue.length) {
      const job = queue.shift();
      if (!job) return;
      queued.delete(job.id);
      slot.busy += 1;
      cutSinceIdle += 1;
      slot.worker.postMessage({
        id: job.id, url: job.photo.u as string, crop: cropBox(job.photo), tile: TILE,
      } satisfies CutRequest);
    }
  }
}

/** How many faces are currently cut and held. */
export function faceCount(): number {
  return done.size;
}

/** Forget every cut face except these, freeing its memory. Returns the ids
 *  that were dropped, so their nodes can go back to being dots. A tile costs
 *  ~36 KB of GPU texture, and the collection is large enough that keeping
 *  every one of them is what makes a tab run out of room. */
export function releaseExcept(keep: Set<string>): string[] {
  const dropped: string[] = [];
  for (const [id, url] of done) {
    if (keep.has(id)) continue;
    URL.revokeObjectURL(url);
    done.delete(id);
    dropped.push(id);
  }
  return dropped;
}

/** Ask for the faces of these nodes, in the order given. */
export function requestFaces(
  ids: string[],
  photoOf: (id: string) => PhotoRef | undefined,
  ready: (id: string, url: string) => void,
  whenIdle?: () => void,
): void {
  onReady = ready;
  onIdle = whenIdle;
  for (const id of ids) {
    if (done.has(id) || failed.has(id) || queued.has(id)) continue;
    const photo = photoOf(id);
    if (!photo?.u) continue;
    queued.add(id);
    queue.push({ id, photo });
  }
  pump();
}
