import { describe, expect, it, vi } from "vitest";
import { flyToNode, formatSimilarity, getSidebarData } from "../src/interactions";
import type { GraphData } from "../src/types";
import type Sigma from "sigma";

function sampleData(): GraphData {
  return {
    meta: { version: "test", count: 3, k: 2 },
    nodes: [
      { id: 0, name: "Alice", x: 0, y: 0, deg: 2, thumb: "thumbs/0.webp", attr: "synthetic", popularity: 10 },
      { id: 1, name: "Bob", x: 10, y: 10, deg: 1, thumb: "thumbs/1.webp", attr: "synthetic", popularity: 8 },
      { id: 2, name: "Carol", x: 20, y: 5, deg: 1, thumb: "thumbs/2.webp", attr: "synthetic", popularity: 5 },
    ],
    edges: [
      [0, 1, 0.912],
      [0, 2, 0.5],
    ],
    similar: {
      "0": [[1, 0.912], [2, 0.5]],
      "1": [[0, 0.912]],
      "2": [[0, 0.5]],
    },
  };
}

describe("flyToNode", () => {
  it("targets the node's normalized display coordinates, not raw graph.json coordinates", () => {
    // Sigma normalizes raw graph coordinates (e.g. our [0,10000] canvas)
    // into a ~[0,1] camera space internally; renderer.getNodeDisplayData
    // returns the already-normalized values. A node at raw (8679.9, 7352.3)
    // on our canvas normalizes to roughly (0.91, 0.74) -- if flyToNode used
    // the raw coordinates instead, the camera would fly ~9500x too far.
    const animate = vi.fn();
    const renderer = {
      getNodeDisplayData: () => ({ x: 0.9104, y: 0.7352 }),
      getCamera: () => ({ animate }),
    } as unknown as Sigma;

    flyToNode(renderer, "5", 500);

    expect(animate).toHaveBeenCalledWith({ x: 0.9104, y: 0.7352, ratio: 0.15 }, { duration: 500 });
  });

  it("throws if the node has no cached display data", () => {
    const renderer = {
      getNodeDisplayData: () => undefined,
      getCamera: () => ({ animate: vi.fn() }),
    } as unknown as Sigma;

    expect(() => flyToNode(renderer, "unknown", 500)).toThrow();
  });
});

describe("formatSimilarity", () => {
  it("formats a cosine weight as a rounded percentage", () => {
    expect(formatSimilarity(0.912)).toBe("91.2%");
    expect(formatSimilarity(0.5)).toBe("50%");
  });
});

describe("getSidebarData", () => {
  it("returns node info plus ranked similar list with formatted percentages", () => {
    const sidebar = getSidebarData(sampleData(), 0);
    expect(sidebar.name).toBe("Alice");
    expect(sidebar.similar).toEqual([
      { id: 1, name: "Bob", percent: "91.2%" },
      { id: 2, name: "Carol", percent: "50%" },
    ]);
  });

  it("throws for an unknown node id", () => {
    expect(() => getSidebarData(sampleData(), 99)).toThrow();
  });
});
