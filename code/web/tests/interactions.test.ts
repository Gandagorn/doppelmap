import { describe, expect, it, vi } from "vitest";
import {
  flyToNode, formatSimilarity, getSidebarData, groupByResemblance, resemblanceBand,
} from "../src/interactions";
import type { GraphData } from "../src/types";
import type Sigma from "sigma";

function sampleData(): GraphData {
  return {
    meta: { version: "test", count: 3, k: 2 },
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
  it("names each band by its score", () => {
    expect(resemblanceBand(0.6)?.label).toBe("Lookalike");
    expect(resemblanceBand(0.3)?.label).toBe("Lookalike");
    expect(resemblanceBand(0.29)?.label).toBe("Looks somewhat alike");
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
