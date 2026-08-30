import { describe, expect, it } from "vitest";
import { searchNames } from "../src/search";
import type { GraphNode } from "../src/types";

function node(id: number, name: string): GraphNode {
  return { id, name, x: 0, y: 0, deg: 0, thumb: "", attr: "" };
}

const NODES = [
  node(0, "Tom Hanks"),
  node(1, "Tom Holland"),
  node(2, "Beyoncé"),
  node(3, "Bad Bunny"),
];

describe("searchNames", () => {
  it("returns empty array for blank query", () => {
    expect(searchNames(NODES, "  ")).toEqual([]);
  });

  it("ranks prefix matches before substring matches", () => {
    const results = searchNames(NODES, "tom");
    expect(results.map((n) => n.name)).toEqual(["Tom Hanks", "Tom Holland"]);
  });

  it("matches substrings not just prefixes", () => {
    const results = searchNames(NODES, "bunny");
    expect(results.map((n) => n.name)).toEqual(["Bad Bunny"]);
  });

  it("is case-insensitive and diacritic-insensitive", () => {
    const results = searchNames(NODES, "beyonce");
    expect(results.map((n) => n.name)).toEqual(["Beyoncé"]);
  });

  it("respects the limit", () => {
    const manyNodes = Array.from({ length: 20 }, (_, i) => node(i, `Tom ${i}`));
    expect(searchNames(manyNodes, "tom", 5)).toHaveLength(5);
  });
});
