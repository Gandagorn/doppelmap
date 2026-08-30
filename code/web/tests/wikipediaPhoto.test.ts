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
          },
        },
      },
    };
    expect(wikipediaInfoFromApiResponse(data)).toEqual({
      photoUrl: "https://upload.wikimedia.org/x.jpg",
      pageUrl: "https://en.wikipedia.org/wiki/Tom_Hanks",
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
    });
  });

  it("returns nulls when the matched page has neither", () => {
    const data = { query: { pages: { "-1": { title: "Nonexistent Person" } } } };
    expect(wikipediaInfoFromApiResponse(data)).toEqual({ photoUrl: null, pageUrl: null });
  });

  it("returns nulls for a malformed or empty response", () => {
    expect(wikipediaInfoFromApiResponse({})).toEqual({ photoUrl: null, pageUrl: null });
    expect(wikipediaInfoFromApiResponse(null)).toEqual({ photoUrl: null, pageUrl: null });
    expect(wikipediaInfoFromApiResponse(undefined)).toEqual({ photoUrl: null, pageUrl: null });
  });
});
