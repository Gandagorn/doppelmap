import { describe, expect, it, beforeEach } from "vitest";
import { creditHtml, faceHtml } from "../src/render";
import { rememberUploadedPhotos } from "../src/photos";
import type { PhotoRef } from "../src/types";

const photo: PhotoRef = {
  f: "Tom Hanks TIFF 2019.jpg",
  b: [0.3, 0.2, 0.6, 0.5],
  a: 1.5,
  c: "Jane Photographer",
  l: "CC BY-SA 4.0",
};

describe("faceHtml", () => {
  beforeEach(() => {
    rememberUploadedPhotos(1, [photo]);
  });

  it("falls back to initials when the person has no photograph", () => {
    const html = faceHtml(9999, "Ada Lovelace", "pair-face", 160);

    expect(html).toContain("face-fallback");
    expect(html).toContain(">AL<");
    expect(html).toContain('aria-label="Ada Lovelace"');
  });

  it("takes at most two initials", () => {
    expect(faceHtml(9999, "Hubert Blaine Wolfeschlegelstein", "c", 80))
      .toContain(">HB<");
  });

  it("escapes a name rather than letting it close the attribute", () => {
    const html = faceHtml(9999, '"><script>alert(1)</script>', "c", 80);

    expect(html).not.toContain("<script>");
  });

  it("lets the container own the size when filling", () => {
    // A hardcoded inline width cannot be overridden by the mobile sheet,
    // which is what rendered a 160px crop inside a 72px frame.
    expect(faceHtml(1, "Tom Hanks", "c", 160, true))
      .toContain("width:100%;height:100%");
    expect(faceHtml(1, "Tom Hanks", "c", 160, false))
      .toContain("width:160px;height:160px");
  });
});

describe("creditHtml", () => {
  it("joins the creator and licence", () => {
    const html = creditHtml(photo);

    expect(html).toContain("Jane Photographer · CC BY-SA 4.0");
  });

  it("names Wikimedia Commons when no creator is recorded", () => {
    expect(creditHtml({ ...photo, c: "" })).toContain("Wikimedia Commons");
  });

  it("omits the separator when there is only one part", () => {
    const html = creditHtml({ ...photo, c: "", l: "" });

    expect(html).toContain("Wikimedia Commons");
    expect(html).not.toContain("·");
  });

  it("escapes a creator name containing markup", () => {
    const html = creditHtml({ ...photo, c: '<img onerror="x">' });

    expect(html).not.toContain("<img");
  });
});
