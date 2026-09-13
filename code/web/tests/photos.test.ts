import { describe, expect, it } from "vitest";
import { faceCropStyle, photoUrl } from "../src/photos";
import type { PhotoRef } from "../src/types";

const photo = (
  b: [number, number, number, number], f = "Tom Hanks 1989.jpg", a = 1
): PhotoRef => ({ f, b, c: "", l: "", a });

/** Pull the numbers back out of the emitted CSS. */
function parse(style: string) {
  const size = /background-size:([\d.]+)%/.exec(style);
  const pos = /background-position:([\d.]+)% ([\d.]+)%/.exec(style);
  return { zoom: Number(size?.[1]), x: Number(pos?.[1]), y: Number(pos?.[2]) };
}

describe("photoUrl", () => {
  it("builds a Special:FilePath link from the bare filename", () => {
    expect(photoUrl(photo([0, 0, 1, 1]), 640)).toBe(
      "https://commons.wikimedia.org/wiki/Special:FilePath/Tom_Hanks_1989.jpg?width=640"
    );
  });

  it("escapes characters that would break the URL", () => {
    // Commons filenames routinely carry brackets, ampersands and accents.
    const url = photoUrl(photo([0, 0, 1, 1], "Beyoncé (crop) & co.jpg"));
    expect(url).toContain("Beyonc%C3%A9");
    expect(url).toContain("%26");
    expect(url).not.toContain(" ");
  });
});

describe("faceCropStyle", () => {
  it("crops with a background rather than a transformed image", () => {
    // A transform escapes the parent's overflow: scaled-up thumbnails
    // spilled across the names next to them. A background cannot.
    const style = faceCropStyle(photo([0.4, 0.4, 0.5, 0.5]));
    expect(style).toContain("background-image:url(");
    expect(style).toContain("background-repeat:no-repeat");
    expect(style).not.toContain("transform");
    expect(style).not.toContain("object-fit");
  });

  it("zooms in on a small face", () => {
    // A face filling a tenth of the frame needs heavy magnification -- the
    // group-shot case that plain centring gets wrong.
    expect(parse(faceCropStyle(photo([0.4, 0.4, 0.5, 0.5]))).zoom).toBeGreaterThan(400);
  });

  it("never zooms below the frame", () => {
    // Under 100% the image would letterbox inside its own element.
    expect(parse(faceCropStyle(photo([0, 0, 1, 1]))).zoom).toBe(100);
  });

  it("follows the face rather than the centre of the image", () => {
    const left = parse(faceCropStyle(photo([0.05, 0.4, 0.15, 0.5])));
    const right = parse(faceCropStyle(photo([0.85, 0.4, 0.95, 0.5])));
    expect(left.x).toBeLessThan(25);
    expect(right.x).toBeGreaterThan(75);
  });

  it("keeps the offsets inside the usable range", () => {
    // Outside 0..100 the image would be pushed off its own frame.
    const boxes: [number, number, number, number][] = [
      [0, 0, 0.05, 0.05], [0.95, 0.95, 1, 1], [0.4, 0, 0.6, 0.2], [0, 0.45, 1, 0.55],
    ];
    for (const b of boxes) {
      const { x, y } = parse(faceCropStyle(photo(b)));
      expect(x).toBeGreaterThanOrEqual(0);
      expect(x).toBeLessThanOrEqual(100);
      expect(y).toBeGreaterThanOrEqual(0);
      expect(y).toBeLessThanOrEqual(100);
    }
  });

  it("survives a degenerate box without producing NaN", () => {
    const { zoom, x, y } = parse(faceCropStyle(photo([0.5, 0.5, 0.5, 0.5])));
    expect(Number.isFinite(zoom)).toBe(true);
    expect(Number.isFinite(x)).toBe(true);
    expect(Number.isFinite(y)).toBe(true);
  });

  it("leaves headroom around the face", () => {
    // Cropped exactly to the detector's box a portrait reads as a mugshot,
    // so it is the padded box that fills the frame.
    const padded = parse(faceCropStyle(photo([0.4, 0.4, 0.6, 0.6]), 0.55));
    const tight = parse(faceCropStyle(photo([0.4, 0.4, 0.6, 0.6]), 0));
    expect(padded.zoom).toBeLessThan(tight.zoom);
  });

  it("asks for a bigger source when the face is small", () => {
    // A face filling a twentieth of the frame needs far more pixels than
    // one filling it entirely, or the crop renders as a smear. This is what
    // group shots looked like before.
    const tiny = faceCropStyle(photo([0.45, 0.45, 0.5, 0.5]), 0.55, 160);
    const full = faceCropStyle(photo([0, 0, 1, 1]), 0.55, 160);
    const widthOf = (s: string) => Number(/width=(\d+)/.exec(s)?.[1]);
    expect(widthOf(tiny)).toBeGreaterThan(widthOf(full) * 4);
  });

  it("keeps the requested width within sane bounds", () => {
    const widthOf = (s: string) => Number(/width=(\d+)/.exec(s)?.[1]);
    // Never so small the crop is mush, never so large it pulls a huge file.
    expect(widthOf(faceCropStyle(photo([0, 0, 1, 1]), 0.55, 36))).toBeGreaterThanOrEqual(320);
    expect(widthOf(faceCropStyle(photo([0.5, 0.5, 0.505, 0.505]), 0.55, 520)))
      .toBeLessThanOrEqual(2400);
  });
});

describe("faceCropStyle quoting", () => {
  it("uses single quotes so it survives inside a style attribute", () => {
    // Emitted into style="...", so a double quote here would terminate the
    // attribute and silently discard the rest of the rule -- which is
    // exactly what happened: every face rendered as an empty box.
    const style = faceCropStyle(photo([0.3, 0.3, 0.5, 0.5]));
    expect(style).toContain("url('");
    expect(style).not.toContain('url("');
    const attr = `<span style="${style}"></span>`;
    expect(attr.match(/"/g)).toHaveLength(2); // only the attribute's own pair
  });
});

describe("faceCropStyle vertical placement", () => {
  it("centres on the face in a tall image", () => {
    // `background-size: Z% auto` scales height by Z/aspect, so the vertical
    // magnification is the zoom divided by the aspect ratio. Multiplying
    // instead pushed the visible band past the face and showed a flat patch
    // of background. Detected faces are near-square in pixels, so a face
    // twice as tall as wide in *fractions* means a 2:1 portrait image.
    const tall = 0.5;  // a 1:2 portrait
    const top = parse(faceCropStyle(photo([0.45, 0.02, 0.55, 0.22], "x.jpg", tall)));
    const bottom = parse(faceCropStyle(photo([0.45, 0.78, 0.55, 0.98], "x.jpg", tall)));
    expect(top.y).toBeLessThan(25);
    expect(bottom.y).toBeGreaterThan(75);
  });

  it("centres on the face in a wide image", () => {
    const wide = 2;    // a 2:1 landscape
    const top = parse(faceCropStyle(photo([0.45, 0.05, 0.65, 0.15], "x.jpg", wide)));
    const bottom = parse(faceCropStyle(photo([0.45, 0.85, 0.65, 0.95], "x.jpg", wide)));
    expect(top.y).toBeLessThan(bottom.y);
  });
});
