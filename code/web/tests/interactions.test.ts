import { describe, expect, it, vi } from "vitest";
import {
  flyToNode, formatSimilarity, getSidebarData, groupByResemblance, rarityNote,
  resemblanceBand,
} from "../src/interactions";
import type { GraphData } from "../src/types";
import type Sigma from "sigma";

function sampleData(): GraphData {
  return {
    meta: {
      version: "test", count: 3, k: 2,
      // Three people make three pairs; tail[i] counts those at or
      // above i/100, so 0.9 and 0.5 both clear 0.50 and only 0.9
      // clears 0.90.
      scale: {
        pairs: 3,
        tail: Array.from({ length: 101 }, (_, i) =>
          i <= 50 ? 2 : i <= 90 ? 1 : 0),
        top: 0.9,
      },
    },
    nodes: [
      { id: 0, name: "Alice", x: 0, y: 0, deg: 2, faces: 12 },
      { id: 1, name: "Bob", x: 10, y: 10, deg: 1, faces: 12 },
      { id: 2, name: "Carol", x: 20, y: 5, deg: 1, faces: 12 },
    ],
    edges: [
      [0, 1, 0.912],
      [0, 2, 0.5],
    ],
    similar: {
      // [otherId, similarity, myPhotoIndex, theirPhotoIndex]
      "0": [[1, 0.912, 2, 3], [2, 0.5, 0, 1]],
      "1": [[0, 0.912, 0, 0]],
      "2": [[0, 0.5, 0, 0]],
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

    flyToNode(renderer, "5");

    expect(animate).toHaveBeenCalledWith({ x: 0.9104, y: 0.7352, ratio: 0.15 }, { duration: 500 });
  });

  it("throws if the node has no cached display data", () => {
    const renderer = {
      getNodeDisplayData: () => undefined,
      getCamera: () => ({ animate: vi.fn() }),
    } as unknown as Sigma;

    expect(() => flyToNode(renderer, "unknown")).toThrow();
  });

  it("offsets the camera so the node lands on the requested screen point", () => {
    // The mobile sidebar covers the bottom half, so the node should end up
    // in the upper half of the canvas rather than dead centre -- which
    // means the camera itself sits *below* the node.
    const animate = vi.fn();
    // Stand-in for Sigma's conversion into its normalized "framed graph"
    // space: 1 unit per 1000px at this zoom, y growing downward like the
    // viewport. viewportToGraph is deliberately given a wildly different
    // scale -- it returns raw graph.json units in the real API, and using
    // it here sent the camera off into nowhere.
    const renderer = {
      getNodeDisplayData: () => ({ x: 0.5, y: 0.5 }),
      getCamera: () => ({ animate, getState: () => ({ x: 0, y: 0, ratio: 1, angle: 0 }) }),
      getDimensions: () => ({ width: 400, height: 800 }),
      viewportToFramedGraph: ({ x, y }: { x: number; y: number }) => ({
        x: x / 1000,
        y: y / 1000,
      }),
      viewportToGraph: ({ x, y }: { x: number; y: number }) => ({ x: x * 25, y: y * 25 }),
    } as unknown as Sigma;

    flyToNode(renderer, "5", { x: 200, y: 200 });

    // Focus is 200px above the viewport centre (400), i.e. -0.2 graph
    // units, so the camera moves +0.2 to push the node up the screen.
    expect(animate).toHaveBeenCalledWith({ x: 0.5, y: 0.7, ratio: 0.15 }, { duration: 500 });
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
      { id: 1, name: "Bob", percent: "91.2%", weight: 0.912, myPhoto: 2, theirPhoto: 3 },
      { id: 2, name: "Carol", percent: "50%", weight: 0.5, myPhoto: 0, theirPhoto: 1 },
    ]);
  });

  it("throws for an unknown node id", () => {
    expect(() => getSidebarData(sampleData(), 99)).toThrow();
  });
});

describe("resemblanceBand", () => {
  it("keeps the strongest pairs on one side of the top cut", () => {
    // The five curated landing pairs, which are shown one after another.
    // They spanned the old 0.30 cut, so paging through them changed the
    // verdict between 30.2% and 29.6%.
    for (const weight of [0.317, 0.344, 0.334, 0.302, 0.295]) {
      expect(resemblanceBand(weight)?.label).toBe("Lookalike");
    }
  });

  it("names each band by its score", () => {
    expect(resemblanceBand(0.6)?.label).toBe("Lookalike");
    expect(resemblanceBand(0.285)?.label).toBe("Lookalike");
    expect(resemblanceBand(0.28)?.label).toBe("Looks somewhat alike");
    expect(resemblanceBand(0.22)?.label).toBe("Looks somewhat alike");
    expect(resemblanceBand(0.21)?.label).toBe("Far resemblance");
    expect(resemblanceBand(0.14)?.label).toBe("Far resemblance");
  });

  it("carries a slug so styling never matches on the wording", () => {
    expect(resemblanceBand(0.4)?.slug).toBe("strong");
    expect(resemblanceBand(0.25)?.slug).toBe("medium");
    expect(resemblanceBand(0.15)?.slug).toBe("faint");
  });

  it("returns null below the floor", () => {
    // Half of all pairs sit between 0.14 and 0.18, where the differences are
    // noise. Showing them as percentages implies a precision that is not there.
    expect(resemblanceBand(0.139)).toBeNull();
    expect(resemblanceBand(0.1)).toBeNull();
    expect(resemblanceBand(0)).toBeNull();
  });
});

describe("groupByResemblance", () => {
  const rows = (...weights: number[]) => weights.map((weight) => ({ weight }));

  it("groups a ranked list into labelled runs, strongest first", () => {
    const groups = groupByResemblance(rows(0.4, 0.28, 0.24, 0.2));
    expect(groups.map((g) => g.band.label)).toEqual([
      "Lookalike", "Looks somewhat alike", "Far resemblance",
    ]);
    expect(groups[1].entries).toHaveLength(2);
  });

  it("drops everything below the floor", () => {
    expect(groupByResemblance(rows(0.25, 0.13, 0.12))).toEqual([
      { band: { label: "Looks somewhat alike", slug: "medium" }, entries: [{ weight: 0.25 }] },
    ]);
  });

  it("returns nothing when no pair clears the floor", () => {
    // No one in the current collection is, but a thinner dataset could be,
    // so the caller must handle it rather than assume every node has someone.
    expect(groupByResemblance(rows(0.13, 0.1))).toEqual([]);
  });
});

describe("getSidebarData and family", () => {
  it("carries the relation through when the two are related", () => {
    const data = sampleData();
    data.similar["0"] = [[1, 0.912, 2, 3, "sibling"], [2, 0.5, 0, 1]];

    const rows = getSidebarData(data, 0).similar;

    expect(rows[0].relation).toBe("sibling");
  });

  it("leaves relation undefined for an ordinary match", () => {
    // Most rows have no fifth element at all -- fifteen per person is the
    // bulk of graph.json, so the common case must not carry the flag.
    expect(getSidebarData(sampleData(), 0).similar[0].relation).toBeUndefined();
  });

  it("still reads a row that has no photo indices", () => {
    const data = sampleData();
    data.similar["0"] = [[1, 0.912] as unknown as (typeof data.similar)["0"][0]];

    const row = getSidebarData(data, 0).similar[0];

    expect(row.myPhoto).toBe(0);
    expect(row.theirPhoto).toBe(0);
    expect(row.relation).toBeUndefined();
  });
});

describe("rarityNote", () => {
  /** tail[i] = pairs scoring at least i/100, so it only ever decreases. */
  function scale(counts: Record<number, number>, pairs = 1_000_000, top = 0.68) {
    let running = pairs;
    const tail = Array.from({ length: 101 }, (_, i) => {
      if (counts[i] !== undefined) running = counts[i];
      return running;
    });
    return { pairs, tail, top };
  }

  it("turns a score into how rare it is", () => {
    const note = rarityNote(0.296, scale({ 29: 100 }));
    expect(note).toContain("1 in 10,000 pairs score this high");
  });

  it("floors to the whole percent below, which understates rarity", () => {
    // 0.296 must be counted against every pair at or above 0.29, not 0.30.
    // Reading the higher bucket would claim the pair is rarer than measured.
    const s = scale({ 29: 100, 30: 10 });
    expect(rarityNote(0.296, s)).toContain("1 in 10,000");
    expect(rarityNote(0.3, s)).toContain("1 in 100,000");
  });

  it("anchors both ends of the scale", () => {
    const note = rarityNote(0.296, scale({ 29: 100 }));
    expect(note).toContain("a random pair scores 0%");
    expect(note).toContain("the map's closest pair, 68%");
  });

  it("rounds away invented precision", () => {
    // 1528626 / 95 is 16090.8; two significant figures is all the floored
    // bucket supports.
    expect(rarityNote(0.296, scale({ 29: 95 }, 1_528_626)))
      .toContain("1 in 16,000 pairs");
  });

  it("survives the top pair, which nothing outranks", () => {
    expect(rarityNote(0.68, scale({ 68: 1 }))).toContain("1 in 1,000,000 pairs");
  });

  it("survives a score off the bottom of the scale", () => {
    expect(rarityNote(-0.2, scale({ 0: 500_000 }))).toContain("1 in 2 pairs");
  });
});
