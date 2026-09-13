import { describe, expect, it } from "vitest";
import { buildGraphology, edgeStrengthScale } from "../src/graphData";
import type { GraphData, GraphEdge } from "../src/types";

function sampleData(): GraphData {
  return {
    meta: { version: "test", count: 3, k: 2 },
    nodes: [
      { id: 0, name: "Alice", x: 0, y: 0, deg: 2, faces: 12 },
      { id: 1, name: "Bob", x: 10, y: 10, deg: 1, faces: 12 },
      { id: 2, name: "Carol", x: 20, y: 5, deg: 1, faces: 12 },
    ],
    edges: [
      [0, 1, 0.9],
      [0, 2, 0.5],
    ],
    similar: {
      "0": [[1, 0.9, 0, 0], [2, 0.5, 0, 0]],
      "1": [[0, 0.9, 0, 0]],
      "2": [[0, 0.5, 0, 0]],
    },
  };
}

describe("buildGraphology", () => {
  it("creates one graph node per data node", () => {
    const graph = buildGraphology(sampleData(), false);
    expect(graph.order).toBe(3);
  });

  it("creates one graph edge per data edge", () => {
    const graph = buildGraphology(sampleData(), false);
    expect(graph.size).toBe(2);
  });

  it("copies node attributes across", () => {
    const graph = buildGraphology(sampleData(), false);
    expect(graph.getNodeAttribute("1", "label")).toBe("Bob");
    expect(graph.getNodeAttribute("1", "deg")).toBe(1);
  });

  it("connects the correct endpoints", () => {
    const graph = buildGraphology(sampleData(), false);
    expect(graph.hasEdge("0", "1")).toBe(true);
    expect(graph.hasEdge("1", "2")).toBe(false);
  });

  it("colors nodes with the light-mode accent when isDark is false", () => {
    const graph = buildGraphology(sampleData(), false);
    expect(graph.getNodeAttribute("0", "color")).toBe("#2a78d6");
  });

  it("colors nodes with the dark-mode accent when isDark is true", () => {
    const graph = buildGraphology(sampleData(), true);
    expect(graph.getNodeAttribute("0", "color")).toBe("#3987e5");
  });

  it("draws the stronger edge more opaque and thicker than the weaker one", () => {
    const graph = buildGraphology(sampleData(), false);
    const strong = graph.getEdgeAttribute("0", "1", "color") as string; // weight 0.9
    const weak = graph.getEdgeAttribute("0", "2", "color") as string; // weight 0.5
    const alphaOf = (rgba: string) => Number(rgba.slice(rgba.lastIndexOf(",") + 1, -1));
    expect(alphaOf(strong)).toBeGreaterThan(alphaOf(weak));
    expect(graph.getEdgeAttribute("0", "1", "size")).toBeGreaterThan(
      graph.getEdgeAttribute("0", "2", "size") as number
    );
  });

  it("sizes nodes smaller than the previous formula, still scaling with degree", () => {
    const graph = buildGraphology(sampleData(), false);
    const degZero = graph.getNodeAttribute("1", "size"); // deg: 1
    const degTwo = graph.getNodeAttribute("0", "size"); // deg: 2
    expect(degZero).toBeLessThan(3); // old formula's minimum was 3 (deg=0)
    expect(degTwo).toBeGreaterThan(degZero); // still grows with degree
  });
});

describe("edgeStrengthScale", () => {
  const edges = (weights: number[]): GraphEdge[] => weights.map((w, i) => [0, i + 1, w]);

  it("spreads a narrow real-world weight band across the full 0..1 range", () => {
    // Mirrors the actual distribution: everything inside ~0.22-0.29, which
    // a fixed 0..1 scale would render as one indistinguishable value.
    const scale = edgeStrengthScale(edges([0.22, 0.24, 0.26, 0.28, 0.29]));
    expect(scale(0.22)).toBeLessThan(0.1);
    expect(scale(0.29)).toBeGreaterThan(0.9);
    expect(scale(0.26)).toBeGreaterThan(0.3);
    expect(scale(0.26)).toBeLessThan(0.7);
  });

  it("keeps one extreme outlier from compressing the rest", () => {
    // 20 ordinary edges packed into 0.20-0.26 plus one 0.99 pair. The
    // sample size matters: percentile bounds can only exclude an outlier
    // when there are enough points for p95 to fall below the maximum, so
    // this mirrors the real datasets (3.5k-17k edges) rather than a
    // handful. The ordinary edges must still spread across the range
    // instead of all collapsing toward 0.
    const ordinary = Array.from({ length: 20 }, (_, i) => 0.2 + i * (0.06 / 19));
    const scale = edgeStrengthScale(edges([...ordinary, 0.99]));
    expect(scale(0.26) - scale(0.2)).toBeGreaterThan(0.5);
    expect(scale(0.99)).toBe(1); // clamped, not off the top of the scale
  });

  it("returns full strength when every edge has the same weight", () => {
    expect(edgeStrengthScale(edges([0.3, 0.3, 0.3]))(0.3)).toBe(1);
  });

  it("handles an empty edge list", () => {
    expect(edgeStrengthScale([])(0.5)).toBe(1);
  });
});
