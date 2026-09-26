import { describe, expect, it } from "vitest";
import { getDisplayMode } from "../src/sigmaSetup";

describe("getDisplayMode", () => {
  it("shows plain dots when zoomed far out", () => {
    expect(getDisplayMode(1.0)).toBe("dot");
  });

  it("shows labels once zoomed past the threshold", () => {
    expect(getDisplayMode(0.3)).toBe("label");
  });

  it("treats the exact threshold as label mode", () => {
    expect(getDisplayMode(0.5)).toBe("label");
  });
});
