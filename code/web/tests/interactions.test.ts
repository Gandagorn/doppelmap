import { describe, expect, it } from "vitest";
import { formatSimilarity, getSidebarData } from "../src/interactions";
import type { GraphData } from "../src/types";

function sampleData(): GraphData {
  return {
    meta: { version: "test", count: 3, k: 2 },
    nodes: [
      { id: 0, name: "Alice", x: 0, y: 0, deg: 2, thumb: "thumbs/0.webp", attr: "synthetic" },
      { id: 1, name: "Bob", x: 10, y: 10, deg: 1, thumb: "thumbs/1.webp", attr: "synthetic" },
      { id: 2, name: "Carol", x: 20, y: 5, deg: 1, thumb: "thumbs/2.webp", attr: "synthetic" },
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
