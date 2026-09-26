import { describe, expect, it } from "vitest";
import {
  DIM_NODE_COLOR, edgeColor, edgeColorForStrength, hexToRgba, nodeColor,
} from "../src/theme";

const luminance = (rgb: string) => {
  const [r, g, b] = rgb.match(/\d+/g)!.map(Number);
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
};

describe("nodeColor", () => {
  it("is a hex colour", () => {
    expect(nodeColor()).toMatch(/^#[0-9a-f]{6}$/i);
  });
});

describe("hexToRgba", () => {
  it("converts a hex colour to rgba at the given alpha", () => {
    expect(hexToRgba("#2a78d6", 0.35)).toBe("rgba(42, 120, 214, 0.35)");
  });
});

describe("edgeColor", () => {
  it("is the accent, made translucent", () => {
    expect(edgeColor()).toBe(hexToRgba(nodeColor(), 0.35));
    expect(edgeColor(0.5)).toBe(hexToRgba(nodeColor(), 0.5));
  });
});

describe("edgeColorForStrength", () => {
  it("draws a stronger edge brighter than a weaker one", () => {
    expect(luminance(edgeColorForStrength(0.9))).toBeGreaterThan(
      luminance(edgeColorForStrength(0.1)));
  });

  it("fades towards the background", () => {
    expect(luminance(edgeColorForStrength(0.9, 0.2))).toBeLessThan(
      luminance(edgeColorForStrength(0.9, 1)));
  });
});

describe("DIM_NODE_COLOR", () => {
  it("is darker than the accent, so dimmed nodes recede", () => {
    expect(luminance(hexToRgba(DIM_NODE_COLOR, 1))).toBeLessThan(
      luminance(hexToRgba(nodeColor(), 1)));
  });
});
