import { describe, expect, it } from "vitest";
import { wikipediaInfoFromApiResponse } from "../src/wikipediaPhoto";

describe("wikipediaInfoFromApiResponse", () => {
  it("extracts the top search result's thumbnail URL and page URL", () => {
    const data = {
      query: {
        pages: {
          "43568": {
            title: "Tom Hanks",
            thumbnail: { source: "https://upload.wikimedia.org/x.jpg", width: 192, height: 266 },
            fullurl: "https://en.wikipedia.org/wiki/Tom_Hanks",
            description: "American actor and filmmaker (born 1956)",
          },
        },
      },
    };
    expect(wikipediaInfoFromApiResponse(data)).toEqual({
      photoUrl: "https://upload.wikimedia.org/x.jpg",
      pageUrl: "https://en.wikipedia.org/wiki/Tom_Hanks",
      description: "American actor and filmmaker (born 1956)",
    });
  });

  it("returns a page URL even when there is no thumbnail", () => {
    const data = {
      query: {
        pages: {
          "1": { title: "Someone", fullurl: "https://en.wikipedia.org/wiki/Someone" },
        },
      },
    };
    expect(wikipediaInfoFromApiResponse(data)).toEqual({
      photoUrl: null,
      pageUrl: "https://en.wikipedia.org/wiki/Someone",
      description: null,
    });
  });

  it("returns nulls when the matched page has neither", () => {
    const data = { query: { pages: { "-1": { title: "Nonexistent Person" } } } };
    expect(wikipediaInfoFromApiResponse(data)).toEqual({
      photoUrl: null,
      pageUrl: null,
      description: null,
    });
  });

  it("returns nulls for a malformed or empty response", () => {
    const empty = { photoUrl: null, pageUrl: null, description: null };
    expect(wikipediaInfoFromApiResponse({})).toEqual(empty);
    expect(wikipediaInfoFromApiResponse(null)).toEqual(empty);
    expect(wikipediaInfoFromApiResponse(undefined)).toEqual(empty);
  });
});
