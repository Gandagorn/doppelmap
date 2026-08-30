import { describe, expect, it } from "vitest";
import { DIM_NODE_COLOR, edgeColor, hexToRgba, nodeColor } from "../src/theme";

describe("nodeColor", () => {
  it("returns the dark-mode accent when dark is preferred", () => {
    expect(nodeColor(true)).toBe("#3987e5");
  });

  it("returns the light-mode accent otherwise", () => {
    expect(nodeColor(false)).toBe("#2a78d6");
  });
});

describe("hexToRgba", () => {
  it("converts a hex color to an rgba string at the given alpha", () => {
    expect(hexToRgba("#2a78d6", 0.35)).toBe("rgba(42, 120, 214, 0.35)");
  });
});

describe("edgeColor", () => {
  it("returns a translucent version of the light accent", () => {
    expect(edgeColor(false)).toBe("rgba(42, 120, 214, 0.35)");
  });

  it("returns a translucent version of the dark accent", () => {
    expect(edgeColor(true)).toBe("rgba(57, 135, 229, 0.35)");
  });

  it("respects a custom alpha", () => {
    expect(edgeColor(false, 0.5)).toBe("rgba(42, 120, 214, 0.5)");
  });
});

describe("DIM_NODE_COLOR", () => {
  it("is a fixed muted color usable in both themes", () => {
    expect(DIM_NODE_COLOR).toBe("#9ca3af");
  });
});
