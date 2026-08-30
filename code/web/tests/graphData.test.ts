import { describe, expect, it } from "vitest";
import { buildGraphology } from "../src/graphData";
import type { GraphData } from "../src/types";

function sampleData(): GraphData {
  return {
    meta: { version: "test", count: 3, k: 2 },
    nodes: [
      { id: 0, name: "Alice", x: 0, y: 0, deg: 2, thumb: "thumbs/0.webp", photo: null, attr: "synthetic" },
      { id: 1, name: "Bob", x: 10, y: 10, deg: 1, thumb: "thumbs/1.webp", photo: null, attr: "synthetic" },
      { id: 2, name: "Carol", x: 20, y: 5, deg: 1, thumb: "thumbs/2.webp", photo: null, attr: "synthetic" },
    ],
    edges: [
      [0, 1, 0.9],
      [0, 2, 0.5],
    ],
    similar: {
      "0": [[1, 0.9], [2, 0.5]],
      "1": [[0, 0.9]],
      "2": [[0, 0.5]],
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
    expect(graph.getNodeAttribute("1", "thumb")).toBe("thumbs/1.webp");
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

  it("colors edges with a translucent accent matching the theme", () => {
    const graph = buildGraphology(sampleData(), false);
    expect(graph.getEdgeAttribute("0", "1", "color")).toBe("rgba(42, 120, 214, 0.35)");
  });
});
