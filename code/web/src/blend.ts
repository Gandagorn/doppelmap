/** Blending two faces into one, in the browser.
 *
 *  A pair of photographs side by side asks the reader to do the comparing.
 *  Blending them does it for them: warp both onto the shape halfway between
 *  the two and average, and what the faces share comes out sharp while what
 *  they do not cancels. When the pair is a real resemblance the result looks
 *  like a person who could be either of them, which is a claim a percentage
 *  cannot make.
 *
 *  Everything happens here rather than in the pipeline because a blend is a
 *  property of a *pair*: 1,701 people make 15,361 openable pairs, so
 *  pre-rendering them would be 184MB of images for a repository that stores
 *  no images at all. Faces are per-person and already being fetched, so
 *  doing the work on open costs nothing but the landmark model.
 *
 *  Three steps, none of which needs a face detector -- photos.json already
 *  stores the box, which is the crop 2d106det expects:
 *
 *    landmarks   2d106det gives 106 points per face. The five points
 *                ArcFace aligns on would be useless here: alignment puts
 *                them at identical coordinates by construction, so there
 *                would be no displacement left to interpolate. The jaw,
 *                brows and lips are what differ.
 *    align       each face is fitted onto one canonical template, so both
 *                land in the same frame at the same scale.
 *    warp        218 triangles from each face onto the intermediate shape,
 *                then mix. At t = 0.5 this is the blend; sweeping t from 0
 *                to 1 is a morph, which is what the card animates.
 */
import { BORDER_POINTS, FACE_TEMPLATE, FACE_TRIANGLES } from "./faceTemplate";
import type { PhotoRef } from "./types";

/** The model's input is fixed at 192 and the crop is built to match. */
const LMK_IN = 192;
const POINTS = 106;
/** Commons width to blend from. The face is a fraction of the frame, so a
 *  thumbnail this size leaves roughly a 300-500px face for a 380px canvas. */
const SOURCE_WIDTH = 900;
const MODEL = "models/2d106det_int8.onnx";

/** Seams. Clipping each triangle to its exact edges leaves a hairline of
 *  background between neighbours once the edges are antialiased, so every
 *  destination triangle is pushed this far out from its own centroid. */
const OVERDRAW = 0.6;

interface Session {
  run(feeds: Record<string, unknown>): Promise<Record<string, { data: Float32Array }>>;
  inputNames: string[];
  outputNames: string[];
}
interface Ort {
  env: { wasm: { wasmPaths: string; numThreads: number } };
  Tensor: new (type: string, data: Float32Array, dims: number[]) => unknown;
  InferenceSession: { create(path: string, opts: object): Promise<Session> };
}

let ortPromise: Promise<Ort> | null = null;
let sessionPromise: Promise<Session> | null = null;

/** The runtime, loaded from our own copy rather than through the bundler.
 *
 *  Vite follows the `import.meta.url` reference inside onnxruntime-web's
 *  bundler build and emits its own 14MB copy of the wasm beside the one
 *  scripts/copy-ort.mjs already places in public/ort/. Importing the URL
 *  keeps the runtime out of the build graph, so it ships exactly once and
 *  at a path this file can predict. */
function runtime(): Promise<Ort> {
  if (!ortPromise) {
    const base = import.meta.env.BASE_URL;
    ortPromise = import(/* @vite-ignore */ `${base}ort/ort.wasm.min.mjs`)
      .then((mod) => {
        const ort = (mod.default ?? mod) as Ort;
        ort.env.wasm.wasmPaths = `${base}ort/`;
        // GitHub Pages sends no cross-origin isolation headers, so
        // SharedArrayBuffer is unavailable and a thread pool would only be
        // created to fall straight back to one thread.
        ort.env.wasm.numThreads = 1;
        return ort;
      })
      .catch((err) => { ortPromise = null; throw err; });
  }
  return ortPromise;
}

/** Whether this browser can blend at all.
 *
 *  Checked rather than assumed so the card can fall back to two photographs
 *  side by side instead of showing a slot that never fills. */
export function blendSupported(): boolean {
  return typeof WebAssembly === "object"
    && typeof createImageBitmap === "function"
    && typeof document.createElement("canvas").getContext === "function";
}

/** The landmark model, loaded once and shared.
 *
 *  Imported dynamically so neither the runtime nor the model is fetched
 *  until someone actually opens a comparison: together they are 15MB, which
 *  no visitor should pay for just to look at the map. */
async function session(): Promise<Session> {
  if (!sessionPromise) {
    sessionPromise = (async () => {
      const ort = await runtime();
      return ort.InferenceSession.create(`${import.meta.env.BASE_URL}${MODEL}`, {
        executionProviders: ["wasm"],
      });
    })().catch((err) => {
      sessionPromise = null;            // let a later open try again
      throw err;
    });
  }
  return sessionPromise;
}

/** The real URL of each thumbnail, which is the only one a canvas may read.
 *
 *  Special:FilePath -- what the rest of the site links to -- answers 302 and
 *  redirects to the image host. The browser applies the CORS check to every
 *  hop of a redirect, and that 302 carries no header, so a cross-origin
 *  fetch of it fails before it ever reaches the image. The API hands back
 *  the final URL instead, takes both titles in one request, and is itself
 *  CORS-enabled with origin=*.
 */
const resolved = new Map<string, string>();

async function thumbUrls(photos: PhotoRef[], width: number): Promise<string[]> {
  const wanted = photos.map((p) => `File:${p.f.replace(/ /g, "_")}`);
  const missing = [...new Set(wanted.filter((t) => !resolved.has(t)))];
  if (missing.length) {
    const query = new URLSearchParams({
      action: "query", format: "json", formatversion: "2", prop: "imageinfo",
      iiprop: "url", iiurlwidth: String(width),
      titles: missing.join("|"), origin: "*",
    });
    const response = await fetch(`https://commons.wikimedia.org/w/api.php?${query}`);
    if (!response.ok) throw new Error(`commons api ${response.status}`);
    const body = await response.json();
    // The API rewrites titles as it sees fit -- spaces, capitalisation,
    // redirects -- so results are matched back through its own mapping
    // rather than by assuming the title came back as it went in.
    const rename = new Map<string, string>();
    for (const n of body.query?.normalized ?? []) rename.set(n.from, n.to);
    for (const r of body.query?.redirects ?? []) rename.set(r.from, r.to);
    const byTitle = new Map<string, string>();
    for (const page of body.query?.pages ?? []) {
      const url = page.imageinfo?.[0]?.thumburl;
      if (url) byTitle.set(page.title, url);
    }
    for (const title of missing) {
      const url = byTitle.get(rename.get(title) ?? title);
      if (url) resolved.set(title, url);
    }
  }
  return wanted.map((title) => {
    const url = resolved.get(title);
    if (!url) throw new Error(`no thumbnail for ${title}`);
    return url;
  });
}

/** Commons throttles bursts; the face worker handles this the same way. */
async function bitmap(url: string): Promise<ImageBitmap> {
  for (let attempt = 0; ; attempt++) {
    const response = await fetch(url, { mode: "cors" });
    if (response.ok) return createImageBitmap(await response.blob());
    if ((response.status !== 429 && response.status < 500) || attempt >= 3) {
      throw new Error(`thumbnail ${response.status}`);
    }
    const after = Number(response.headers.get("Retry-After"));
    await new Promise((r) => setTimeout(r, after > 0 ? after * 1000 : 400 * 2 ** attempt));
  }
}

/** 106 landmarks for the face in this photo, in the bitmap's own pixels. */
async function landmarks(image: ImageBitmap, box: PhotoRef["b"]): Promise<Float64Array> {
  const [x1, y1, x2, y2] = [box[0] * image.width, box[1] * image.height,
                            box[2] * image.width, box[3] * image.height];
  const cx = (x1 + x2) / 2;
  const cy = (y1 + y2) / 2;
  // The crop the model was trained on: centred on the box, scaled so the
  // longer side fills two thirds of the input.
  const scale = LMK_IN / (Math.max(x2 - x1, y2 - y1) * 1.5);
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = LMK_IN;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  if (!ctx) throw new Error("no 2d context");
  ctx.setTransform(scale, 0, 0, scale, -scale * cx + LMK_IN / 2, -scale * cy + LMK_IN / 2);
  ctx.drawImage(image, 0, 0);
  ctx.setTransform(1, 0, 0, 1, 0, 0);

  const { data } = ctx.getImageData(0, 0, LMK_IN, LMK_IN);
  const plane = LMK_IN * LMK_IN;
  const input = new Float32Array(3 * plane);
  for (let i = 0, p = 0; i < plane; i++, p += 4) {
    input[i] = data[p];                 // the model takes raw 0..255, RGB
    input[plane + i] = data[p + 1];
    input[2 * plane + i] = data[p + 2];
  }

  const ort = await runtime();
  const run = await session();
  const out = await run.run({
    [run.inputNames[0]]: new ort.Tensor("float32", input, [1, 3, LMK_IN, LMK_IN]),
  });
  const raw = out[run.outputNames[0]].data;

  // The model answers in -1..1 over the crop; undo the crop transform to get
  // back to the bitmap's coordinates.
  const points = new Float64Array(POINTS * 2);
  for (let i = 0; i < POINTS; i++) {
    const x = (raw[i * 2] + 1) * (LMK_IN / 2);
    const y = (raw[i * 2 + 1] + 1) * (LMK_IN / 2);
    points[i * 2] = (x - LMK_IN / 2) / scale + cx;
    points[i * 2 + 1] = (y - LMK_IN / 2) / scale + cy;
  }
  return points;
}

/** The least-squares rotation, uniform scale and shift taking `from` onto
 *  `to`, as a canvas transform.
 *
 *  Closed form rather than an SVD: a 2D similarity has only four unknowns,
 *  and solving them directly also rules out the reflection an SVD would
 *  happily return for a face that is nearly symmetric.
 */
export function similarity(
  from: Float64Array, to: Float64Array, n: number
): DOMMatrix2DInit {
  let mx = 0, my = 0, tx = 0, ty = 0;
  for (let i = 0; i < n; i++) {
    mx += from[i * 2]; my += from[i * 2 + 1];
    tx += to[i * 2]; ty += to[i * 2 + 1];
  }
  mx /= n; my /= n; tx /= n; ty /= n;

  let dot = 0, cross = 0, norm = 0;
  for (let i = 0; i < n; i++) {
    const x = from[i * 2] - mx, y = from[i * 2 + 1] - my;
    const u = to[i * 2] - tx, v = to[i * 2 + 1] - ty;
    dot += x * u + y * v;
    cross += x * v - y * u;
    norm += x * x + y * y;
  }
  const a = dot / Math.max(norm, 1e-9);
  const b = cross / Math.max(norm, 1e-9);
  return { a, b, c: -b, d: a, e: tx - (a * mx - b * my), f: ty - (b * mx + a * my) };
}

/** One face, drawn onto the shared template frame, with its own landmarks
 *  carried into the same coordinates. */
interface Aligned {
  canvas: HTMLCanvasElement;
  /** 114 points: the 106 landmarks, then the canvas corners and midpoints. */
  shape: Float64Array;
}

async function align(photo: PhotoRef, url: string, size: number): Promise<Aligned> {
  const image = await bitmap(url);
  const found = await landmarks(image, photo.b);

  const target = new Float64Array(POINTS * 2);
  for (let i = 0; i < POINTS * 2; i++) target[i] = FACE_TEMPLATE[i] * size;
  const m = similarity(found, target, POINTS);

  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = size;
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("no 2d context");
  ctx.setTransform(m.a!, m.b!, m.c!, m.d!, m.e!, m.f!);
  ctx.drawImage(image, 0, 0);
  image.close();

  // Where this face's landmarks actually landed. They sit near the template
  // but not on it, and that difference is the face shape the blend mixes.
  const shape = new Float64Array((POINTS + 8) * 2);
  for (let i = 0; i < POINTS; i++) {
    const x = found[i * 2], y = found[i * 2 + 1];
    shape[i * 2] = m.a! * x + m.c! * y + m.e!;
    shape[i * 2 + 1] = m.b! * x + m.d! * y + m.f!;
  }
  for (let i = 0; i < 8; i++) {
    shape[(POINTS + i) * 2] = BORDER_POINTS[i * 2] * size;
    shape[(POINTS + i) * 2 + 1] = BORDER_POINTS[i * 2 + 1] * size;
  }
  return { canvas, shape };
}

/** Put the second face on the first one's colour footing.
 *
 *  Two photographs of the same person under different light average into
 *  something that looks like neither, and the pairs here are routinely one
 *  cold daylight shot against one warm interior one. Matching the mean and
 *  spread of each channel costs one pass and fixes most of it.
 *
 *  Statistics are taken from the middle of the frame only. The edges are
 *  background -- a green hedge behind one face and a black studio behind
 *  the other -- and letting those set the correction would drag the skin
 *  the wrong way. */
function matchColour(target: Aligned, reference: Aligned, size: number): void {
  const ctxT = target.canvas.getContext("2d", { willReadFrequently: true });
  const ctxR = reference.canvas.getContext("2d", { willReadFrequently: true });
  if (!ctxT || !ctxR) return;
  const inset = Math.round(size * 0.2);
  const span = size - inset * 2;
  const stats = (data: Uint8ClampedArray) => {
    const sum = [0, 0, 0], sq = [0, 0, 0];
    for (let i = 0; i < data.length; i += 4) {
      for (let c = 0; c < 3; c++) { sum[c] += data[i + c]; sq[c] += data[i + c] ** 2; }
    }
    const n = data.length / 4;
    return [0, 1, 2].map((c) => {
      const mean = sum[c] / n;
      return { mean, sd: Math.sqrt(Math.max(sq[c] / n - mean * mean, 1)) };
    });
  };
  const want = stats(ctxR.getImageData(inset, inset, span, span).data);
  const have = stats(ctxT.getImageData(inset, inset, span, span).data);

  const whole = ctxT.getImageData(0, 0, size, size);
  const px = whole.data;
  for (let c = 0; c < 3; c++) {
    const gain = want[c].sd / have[c].sd;
    const shift = want[c].mean - have[c].mean * gain;
    for (let i = c; i < px.length; i += 4) px[i] = px[i] * gain + shift;
  }
  ctxT.putImageData(whole, 0, 0);
}

/** Push a triangle's corners out from its centroid, to hide the seams. */
function spread(p: number[], by: number): number[] {
  const cx = (p[0] + p[2] + p[4]) / 3;
  const cy = (p[1] + p[3] + p[5]) / 3;
  const out = p.slice();
  for (let i = 0; i < 3; i++) {
    const dx = p[i * 2] - cx, dy = p[i * 2 + 1] - cy;
    const len = Math.hypot(dx, dy) || 1;
    out[i * 2] += (dx / len) * by;
    out[i * 2 + 1] += (dy / len) * by;
  }
  return out;
}

/** Redraw one aligned face with its landmarks moved onto `to`. */
function warp(ctx: CanvasRenderingContext2D, face: Aligned, to: Float64Array): void {
  const from = face.shape;
  for (let t = 0; t < FACE_TRIANGLES.length; t += 3) {
    const i = FACE_TRIANGLES[t], j = FACE_TRIANGLES[t + 1], k = FACE_TRIANGLES[t + 2];
    const s = [from[i * 2], from[i * 2 + 1], from[j * 2], from[j * 2 + 1],
               from[k * 2], from[k * 2 + 1]];
    const d = spread([to[i * 2], to[i * 2 + 1], to[j * 2], to[j * 2 + 1],
                      to[k * 2], to[k * 2 + 1]], OVERDRAW);

    // The affine carrying the source triangle onto the destination one.
    const sx1 = s[2] - s[0], sy1 = s[3] - s[1];
    const sx2 = s[4] - s[0], sy2 = s[5] - s[1];
    const det = sx1 * sy2 - sx2 * sy1;
    if (Math.abs(det) < 1e-6) continue;       // collinear after clamping
    const dx1 = d[2] - d[0], dy1 = d[3] - d[1];
    const dx2 = d[4] - d[0], dy2 = d[5] - d[1];
    const a = (dx1 * sy2 - dx2 * sy1) / det;
    const b = (dy1 * sy2 - dy2 * sy1) / det;
    const c = (dx2 * sx1 - dx1 * sx2) / det;
    const dd = (dy2 * sx1 - dy1 * sx2) / det;

    ctx.save();
    ctx.beginPath();
    ctx.moveTo(d[0], d[1]);
    ctx.lineTo(d[2], d[3]);
    ctx.lineTo(d[4], d[5]);
    ctx.closePath();
    ctx.clip();
    ctx.transform(a, b, c, dd, d[0] - (a * s[0] + c * s[1]), d[1] - (b * s[0] + dd * s[1]));
    ctx.drawImage(face.canvas, 0, 0);
    ctx.restore();
  }
}

/** Fade the frame away outside an oval, so only the faces are compared.
 *
 *  Backgrounds clash long before faces do -- a blue stage light against a
 *  black studio -- and that mismatch is most of what makes a composite read
 *  as two photographs stuck together. Left as alpha rather than filled with
 *  a colour, so the card's own background shows through in either theme. */
function vignette(ctx: CanvasRenderingContext2D, size: number): void {
  const rx = size * 0.46, ry = size * 0.52;
  ctx.save();
  ctx.globalCompositeOperation = "destination-in";
  ctx.translate(size / 2, size * 0.50);
  ctx.scale(1, ry / rx);
  const grad = ctx.createRadialGradient(0, 0, 0, 0, 0, rx * 1.12);
  grad.addColorStop(0, "rgba(0,0,0,1)");
  grad.addColorStop(0.86 / 1.12, "rgba(0,0,0,1)");
  grad.addColorStop(1, "rgba(0,0,0,0)");
  ctx.fillStyle = grad;
  ctx.fillRect(-size, -size, size * 2, size * 2);
  ctx.restore();
}

/** Two faces, prepared once, renderable at any point between them. */
export interface Blend {
  /** t = 0 is the first face, 1 the second, 0.5 the blend. */
  render(t: number): HTMLCanvasElement;
  readonly size: number;
}

/** Fetch both photographs, find their landmarks, and return a renderer.
 *
 *  The expensive half -- two downloads and two model runs -- happens here,
 *  once. Each frame after that is 436 triangles of canvas work, fast enough
 *  for the card to animate between the two faces. */
export async function prepareBlend(a: PhotoRef, b: PhotoRef, size: number): Promise<Blend> {
  // One request resolves both thumbnails, and it overlaps the runtime
  // download rather than queueing behind it.
  const urls = await thumbUrls([a, b], SOURCE_WIDTH);
  const [faceA, faceB] = await Promise.all([
    align(a, urls[0], size), align(b, urls[1], size),
  ]);
  matchColour(faceB, faceA, size);
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = size;
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("no 2d context");

  const mid = new Float64Array(faceA.shape.length);
  return {
    size,
    render(t: number) {
      const k = Math.min(1, Math.max(0, t));
      for (let i = 0; i < mid.length; i++) {
        mid[i] = faceA.shape[i] * (1 - k) + faceB.shape[i] * k;
      }
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.clearRect(0, 0, size, size);
      warp(ctx, faceA, mid);
      // Drawing the second at alpha k over the first is exactly the mix
      // (1 - k)·A + k·B, so t also controls which face dominates.
      ctx.globalAlpha = k;
      warp(ctx, faceB, mid);
      ctx.globalAlpha = 1;
      vignette(ctx, size);
      return canvas;
    },
  };
}
