/** HTML fragments the sidebar and comparison view are built from. */
import { escapeHtml } from "./interactions";
import { faceCropStyle, photosFor } from "./photos";
import type { PhotoRef } from "./types";

export function faceHtml(
  id: number, name: string, cls: string, width: number, fill = false
): string {
  const photo = photosFor(id)?.[0];
  // `fill` lets the container own the size.
  const size = fill ? "width:100%;height:100%" : `width:${width}px;height:${width}px`;
  if (!photo) {
    const initials = name.split(" ").map((w) => w[0]).join("").slice(0, 2);
    return `<span class="${cls} face-fallback" style="${size}"
                  aria-label="${escapeHtml(name)}">${escapeHtml(initials)}</span>`;
  }
  return `<span class="${cls}" style="${faceCropStyle(photo, 0.55, width)}${size}"
                role="img" aria-label="${escapeHtml(name)}"></span>`;
}


export function creditHtml(photo: PhotoRef, cls = "credit"): string {
  const text = [photo.c || "Wikimedia Commons", photo.l].filter(Boolean).join(" · ");
  return `<span class="${cls} credit-mark">
    <button type="button" class="credit-btn" aria-label="Photo credit: ${escapeHtml(text)}"
            title="${escapeHtml(text)}">&#9432;</button>
    <span class="credit-text">${escapeHtml(text)}</span>
  </span>`;
}
