/** Cuts one face tile: fetch the thumbnail, crop to the box, encode a tile.
 *
 *  In a worker because the canvas work, not the network, is what limits how
 *  many faces can be prepared at once: on the main thread the fetches came
 *  back in ~90ms each but only about three were ever in flight.
 */
export interface CutRequest {
  id: string;
  url: string;
  crop: { x: number; y: number; size: number };
  tile: number;
}

export interface CutResult {
  id: string;
  blob?: Blob;
  error?: string;
}

const CACHE = "doppelmap-faces-v2";

/** Wikimedia answers 429 when asked for too much at once, and a face that
 *  gave up on one would never appear. Waits as asked and tries again. */
async function fetchThumbnail(url: string): Promise<Blob> {
  for (let attempt = 0; ; attempt++) {
    const response = await fetch(url, { mode: "cors" });
    if (response.ok) return response.blob();
    const retryable = response.status === 429 || response.status >= 500;
    if (!retryable || attempt >= 3) throw new Error(`thumbnail ${response.status}`);
    const after = Number(response.headers.get("Retry-After"));
    const wait = Number.isFinite(after) && after > 0
      ? after * 1000
      : 500 * 2 ** attempt + Math.random() * 400;
    await new Promise((r) => setTimeout(r, wait));
  }
}

async function cut(req: CutRequest): Promise<Blob> {
  const store = "caches" in self ? await caches.open(CACHE).catch(() => undefined) : undefined;
  const hit = await store?.match(req.url).then((r) => r?.blob()).catch(() => undefined);
  if (hit) return hit;

  const full = await createImageBitmap(await fetchThumbnail(req.url));
  const { x, y, size } = req.crop;
  const sx = x * full.width;
  const sy = y * full.height;
  const sw = Math.max(1, size * full.width);
  const sh = Math.max(1, Math.min(size * full.width, full.height - sy));
  const bitmap = await createImageBitmap(full, sx, sy, sw, sh, {
    resizeWidth: req.tile, resizeHeight: req.tile, resizeQuality: "medium",
  });
  full.close();

  const canvas = new OffscreenCanvas(req.tile, req.tile);
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("no 2d context");
  ctx.drawImage(bitmap, 0, 0);
  bitmap.close();
  // A rim keeps overlapping faces apart, and off the edges crossing behind them.
  ctx.strokeStyle = "rgba(255, 255, 255, 0.5)";
  ctx.lineWidth = 6;
  ctx.beginPath();
  ctx.arc(req.tile / 2, req.tile / 2, req.tile / 2 - 3, 0, Math.PI * 2);
  ctx.stroke();

  const blob = await canvas.convertToBlob({ type: "image/webp", quality: 0.82 });
  void store?.put(req.url, new Response(blob.slice(), {
    headers: { "Content-Type": "image/webp" },
  })).catch(() => undefined);
  return blob;
}

self.onmessage = async (event: MessageEvent<CutRequest>) => {
  const req = event.data;
  try {
    const blob = await cut(req);
    (self as unknown as Worker).postMessage({ id: req.id, blob } satisfies CutResult);
  } catch (err) {
    (self as unknown as Worker).postMessage({ id: req.id, error: String(err) } satisfies CutResult);
  }
};
