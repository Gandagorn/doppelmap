import type { PhotoRef } from "./types";

/** The square of the source image that shows the face, in 0..1 fractions. */
export function faceSourceRect(
  photo: PhotoRef, padding = 0.55
): { x: number; y: number; w: number; h: number } {
  const [x1, y1, x2, y2] = photo.b;
  const aspect = photo.a > 0 ? photo.a : 1;
  // Pixel-proportional units: height 1, width = aspect.
  const width = Math.max(x2 - x1, 0.001) * aspect;
  const height = Math.max(y2 - y1, 0.001);
  const side = Math.min(Math.max(width, height) * (1 + padding), aspect, 1);
  const centreX = (x1 + (x2 - x1) / 2) * aspect;
  const centreY = y1 + (y2 - y1) / 2;
  const x = Math.min(Math.max(centreX - side / 2, 0), aspect - side);
  const y = Math.min(Math.max(centreY - side / 2, 0), 1 - side);
  // Back to fractions of the image, which is what drawImage wants -- and the two axes
  // differ, because one fraction of width is not one of height.
  return { x: x / aspect, y, w: side / aspect, h: side };
}

const API = "https://commons.wikimedia.org/w/api.php";

/** A thumbnail URL a canvas is allowed to read. */
async function corsThumbUrls(
  names: string[], width: number
): Promise<Map<string, string>> {
  const titles = names.map((n) => `File:${n}`);
  const params = new URLSearchParams({
    action: "query", format: "json", origin: "*", prop: "imageinfo",
    iiprop: "url", iiurlwidth: String(width), titles: titles.join("|"),
  });
  const response = await fetch(`${API}?${params}`);
  const body = await response.json();
  const out = new Map<string, string>();
  for (const page of Object.values(body?.query?.pages ?? {}) as {
    title?: string; imageinfo?: { thumburl?: string }[];
  }[]) {
    const url = page.imageinfo?.[0]?.thumburl;
    if (page.title && url) out.set(page.title.replace(/^File:/, ""), url);
  }
  return out;
}

function loadImage(url: string, crossOrigin: boolean): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    if (crossOrigin) img.crossOrigin = "anonymous";
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error(`could not load ${url}`));
    img.src = url;
  });
}

export interface CardSide {
  name: string;
  photo: PhotoRef;
}

export interface CardOptions {
  a: CardSide;
  b: CardSide;
  percent: string;
  band: string;
  /** Painted for dark mode when the page is in it. */
  dark: boolean;
}

const W = 1200;
const H = 630;          // the aspect every link preview crops to
const FACE = 340;
// Wide enough for the percentage to sit between the faces rather than across them: at
// 62px "46.4%" is about 200px, and this leaves 260.
const GAP = 130;

/** Draws the comparison as a picture that can be shared anywhere. */
export async function renderShareCard(options: CardOptions): Promise<Blob> {
  const { a, b, percent, band, dark } = options;
  const canvas = document.createElement("canvas");
  canvas.width = W;
  canvas.height = H;
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("no 2d context");

  const ink = dark ? "#e8edf5" : "#1a2233";
  const muted = dark ? "#93a1b8" : "#64748b";
  ctx.fillStyle = dark ? "#0b1018" : "#f6f9fc";
  ctx.fillRect(0, 0, W, H);

  // Commons filenames need a CORS-capable URL; a blob from an upload is already same-
  // origin and must not ask for one.
  const remote = [a, b].filter((s) => !s.photo.f.startsWith("blob:"));
  const urls = remote.length
    ? await corsThumbUrls(remote.map((s) => s.photo.f), 800).catch(() => new Map())
    : new Map<string, string>();

  const sides = [a, b];
  const left = [W / 2 - FACE - GAP, W / 2 + GAP];
  for (let i = 0; i < sides.length; i++) {
    const side = sides[i];
    const isBlob = side.photo.f.startsWith("blob:");
    const url = isBlob ? side.photo.f : urls.get(side.photo.f);
    const x = left[i];
    const y = 120;

    ctx.save();
    roundedRect(ctx, x, y, FACE, FACE, 24);
    ctx.clip();
    ctx.fillStyle = dark ? "#1b2432" : "#e6edf6";
    ctx.fillRect(x, y, FACE, FACE);
    if (url) {
      try {
        const img = await loadImage(url, !isBlob);
        const rect = faceSourceRect(side.photo);
        ctx.drawImage(
          img,
          rect.x * img.naturalWidth, rect.y * img.naturalHeight,
          rect.w * img.naturalWidth, rect.h * img.naturalHeight,
          x, y, FACE, FACE
        );
      } catch {
        // A missing picture should still produce a card, just a plainer one.
      }
    }
    ctx.restore();

    ctx.fillStyle = ink;
    ctx.font = "600 30px Inter, system-ui, sans-serif";
    ctx.textAlign = "center";
    fitText(ctx, side.name, x + FACE / 2, y + FACE + 48, FACE + 40);
  }

  ctx.textAlign = "center";
  ctx.fillStyle = dark ? "#6aa5ee" : "#2a78d6";
  ctx.font = "700 62px Inter, system-ui, sans-serif";
  ctx.fillText(percent, W / 2, 300);
  ctx.fillStyle = muted;
  ctx.font = "600 20px Inter, system-ui, sans-serif";
  ctx.fillText(band.toUpperCase(), W / 2, 340);

  ctx.fillStyle = muted;
  ctx.font = "400 22px Inter, system-ui, sans-serif";
  ctx.fillText("gandagorn.github.io/doppelmap", W / 2, H - 38);

  return new Promise((resolve, reject) =>
    canvas.toBlob(
      (blob) => (blob ? resolve(blob) : reject(new Error("could not render"))),
      "image/png"
    ));
}

function roundedRect(
  ctx: CanvasRenderingContext2D, x: number, y: number,
  w: number, h: number, r: number
) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

/** Shrinks a long name rather than letting it run into the other face. */
function fitText(
  ctx: CanvasRenderingContext2D, text: string, x: number, y: number, max: number
) {
  let size = 30;
  ctx.font = `600 ${size}px Inter, system-ui, sans-serif`;
  while (ctx.measureText(text).width > max && size > 16) {
    size -= 2;
    ctx.font = `600 ${size}px Inter, system-ui, sans-serif`;
  }
  ctx.fillText(text, x, y);
}
