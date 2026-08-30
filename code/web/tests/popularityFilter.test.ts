import { describe, expect, it } from "vitest";
import { percentileThreshold } from "../src/popularityFilter";

describe("percentileThreshold", () => {
  it("returns the minimum value at percentile 0 (show everyone)", () => {
    expect(percentileThreshold([5, 1, 3, 2, 4], 0)).toBe(1);
  });

  it("returns the maximum value at percentile 100 (show only the top node)", () => {
    expect(percentileThreshold([5, 1, 3, 2, 4], 100)).toBe(5);
  });

  it("returns the median at the 50th percentile", () => {
    expect(percentileThreshold([5, 1, 3, 2, 4], 50)).toBe(3);
  });

  it("returns 0 for an empty list", () => {
    expect(percentileThreshold([], 50)).toBe(0);
  });

  it("clamps percentile values outside [0, 100]", () => {
    const values = [5, 1, 3, 2, 4];
    expect(percentileThreshold(values, -10)).toBe(percentileThreshold(values, 0));
    expect(percentileThreshold(values, 150)).toBe(percentileThreshold(values, 100));
  });

  it("handles a single-value list", () => {
    expect(percentileThreshold([42], 0)).toBe(42);
    expect(percentileThreshold([42], 100)).toBe(42);
  });
});
