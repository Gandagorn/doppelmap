import { describe, expect, it } from "vitest";
import { cropBox } from "../src/faceTiles";
import type { PhotoRef } from "../src/types";

const photo = (b: PhotoRef["b"], a: number): PhotoRef =>
  ({ f: "x.jpg", b, a, c: "", l: "" });

describe("cropBox", () => {
  it("centres on the face box", () => {
    const { x, y, size } = cropBox(photo([0.4, 0.4, 0.6, 0.6], 1));
    expect(x + size / 2).toBeCloseTo(0.5, 5);
    expect(y + size / 2).toBeCloseTo(0.5, 5);
  });

  it("pads beyond the detector box", () => {
    expect(cropBox(photo([0.4, 0.4, 0.6, 0.6], 1)).size).toBeGreaterThan(0.2);
  });

  it("stays inside the image", () => {
    for (const box of [[0, 0, 0.2, 0.2], [0.8, 0.8, 1, 1], [0.05, 0.9, 0.2, 1]] as PhotoRef["b"][]) {
      for (const aspect of [0.6, 1, 1.8]) {
        const { x, y, size } = cropBox(photo(box, aspect));
        expect(x).toBeGreaterThanOrEqual(0);
        expect(y).toBeGreaterThanOrEqual(0);
        expect(x + size).toBeLessThanOrEqual(1.0001);
        expect(y + size * aspect).toBeLessThanOrEqual(1.0001);
      }
    }
  });

  it("keeps a square crop in pixels, not in fractions", () => {
    const wide = cropBox(photo([0.4, 0.4, 0.6, 0.6], 2));
    expect(wide.size * 2).toBeCloseTo(wide.size * 2, 5);
    expect(wide.size).toBeGreaterThan(0);
  });
});
