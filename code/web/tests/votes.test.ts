import { describe, expect, it } from "vitest";
import { describeTally, orderPair } from "../src/votes";

describe("orderPair", () => {
  it("puts the pair in the order the table stores it", () => {
    // The table enforces person_a < person_b, so A/B and B/A have to become
    // the same row -- otherwise one pair collects two separate tallies.
    expect(orderPair("Q2263", "Q42")).toEqual(["Q2263", "Q42"]);
    expect(orderPair("Q42", "Q2263")).toEqual(["Q2263", "Q42"]);
  });

  it("orders the same way whichever side asks", () => {
    const pairs: [string, string][] = [["Q1", "Q999"], ["Q10", "Q9"], ["Q5", "Q5000"]];
    for (const [a, b] of pairs) {
      expect(orderPair(a, b)).toEqual(orderPair(b, a));
    }
  });
});

describe("describeTally", () => {
  it("reports the share that agrees", () => {
    expect(describeTally({ yes: 17, no: 8, total: 25 })).toBe("68% of 25 people say yes");
  });

  it("does not say '1 people'", () => {
    expect(describeTally({ yes: 1, no: 0, total: 1 })).toBe("100% of 1 person say yes");
  });

  it("invites the first vote rather than showing 0%", () => {
    // 0% of 0 is not a fact about the pair, it is a fact about the table.
    expect(describeTally({ yes: 0, no: 0, total: 0 })).toBe("No votes yet — be the first.");
  });
});
