import { describe, expect, it } from "vitest";
import { photoUrlFromApiResponse } from "../src/wikipediaPhoto";

describe("photoUrlFromApiResponse", () => {
  it("extracts the top search result's thumbnail URL", () => {
    const data = {
      query: {
        pages: {
          "12345": {
            title: "Tom Hanks",
            thumbnail: { source: "https://upload.wikimedia.org/x.jpg", width: 192, height: 256 },
          },
        },
      },
    };
    expect(photoUrlFromApiResponse(data)).toBe("https://upload.wikimedia.org/x.jpg");
  });

  it("returns null when the matched page has no thumbnail", () => {
    const data = { query: { pages: { "-1": { title: "Nonexistent Person" } } } };
    expect(photoUrlFromApiResponse(data)).toBeNull();
  });

  it("returns null for a malformed or empty response", () => {
    expect(photoUrlFromApiResponse({})).toBeNull();
    expect(photoUrlFromApiResponse(null)).toBeNull();
    expect(photoUrlFromApiResponse(undefined)).toBeNull();
  });
});
