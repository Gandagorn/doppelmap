import { describe, expect, it } from "vitest";
import { similarity } from "../src/blend";

/** Four points in a rough face arrangement: two eyes, nose, chin. */
const SHAPE = new Float64Array([30, 40, 70, 40, 50, 60, 50, 85]);
const N = 4;

/** Apply a canvas transform the way the alignment step does. */
function apply(m: DOMMatrix2DInit, pts: Float64Array): Float64Array {
  const out = new Float64Array(pts.length);
  for (let i = 0; i < pts.length; i += 2) {
    out[i] = m.a! * pts[i] + m.c! * pts[i + 1] + m.e!;
    out[i + 1] = m.b! * pts[i] + m.d! * pts[i + 1] + m.f!;
  }
  return out;
}

function transform(scale: number, degrees: number, dx: number, dy: number) {
  const r = (degrees * Math.PI) / 180;
  return {
    a: scale * Math.cos(r), b: scale * Math.sin(r),
    c: -scale * Math.sin(r), d: scale * Math.cos(r),
    e: dx, f: dy,
  };
}

describe("similarity", () => {
  it("is the identity when a shape is fitted to itself", () => {
    const m = similarity(SHAPE, SHAPE, N);
    expect(m.a).toBeCloseTo(1, 10);
    expect(m.b).toBeCloseTo(0, 10);
    expect(m.e).toBeCloseTo(0, 10);
    expect(m.f).toBeCloseTo(0, 10);
  });

  it("recovers a scale, rotation and shift exactly", () => {
    const target = apply(transform(1.7, 23, 12, -40), SHAPE);
    const got = apply(similarity(SHAPE, target, N), SHAPE);
    for (let i = 0; i < target.length; i++) expect(got[i]).toBeCloseTo(target[i], 8);
  });

  it("stays a similarity: no shear, uniform scale", () => {
    // A noisy target cannot be matched exactly, and the fit must not reach
    // for a general affine to try -- the warp afterwards assumes the face
    // was only moved, scaled and turned.
    const target = apply(transform(1.3, -14, 5, 9), SHAPE);
    for (let i = 0; i < target.length; i++) target[i] += (i % 3) - 1;
    const m = similarity(SHAPE, target, N);
    expect(m.d).toBeCloseTo(m.a!, 12);          // equal diagonal
    expect(m.c).toBeCloseTo(-m.b!, 12);         // antisymmetric off-diagonal
  });

  it("never answers with a reflection", () => {
    // A face is nearly symmetric, so a least-squares fit that allowed
    // reflection could silently mirror one of the two and the blend would
    // average a left cheek onto a right one.
    const mirrored = new Float64Array(SHAPE);
    for (let i = 0; i < mirrored.length; i += 2) mirrored[i] = 100 - mirrored[i];
    const m = similarity(SHAPE, mirrored, N);
    expect(m.a! * m.d! - m.b! * m.c!).toBeGreaterThan(0);   // positive determinant
  });

  it("is unmoved by the order the points arrive in", () => {
    const target = apply(transform(0.8, 61, -3, 7), SHAPE);
    const swap = (p: Float64Array) =>
      new Float64Array([p[2], p[3], p[0], p[1], p[6], p[7], p[4], p[5]]);
    const a = similarity(SHAPE, target, N);
    const b = similarity(swap(SHAPE), swap(target), N);
    expect(b.a).toBeCloseTo(a.a!, 10);
    expect(b.e).toBeCloseTo(a.e!, 10);
  });
});
