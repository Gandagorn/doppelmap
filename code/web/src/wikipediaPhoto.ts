// Live, on-demand lookup of a real profile photo from Wikipedia -- called
// when a node's sidebar opens, not baked into graph.json at dataset-build
// time, so dataset size never bounds how long generation takes. Wikipedia's
// API supports direct browser calls via `origin=*` (CORS), so this needs no
// server of our own.
const THUMB_WIDTH = 192; // 2x the 96px display size, for retina

export function photoUrlFromApiResponse(data: unknown): string | null {
  const pages = (data as { query?: { pages?: Record<string, unknown> } } | null | undefined)
    ?.query?.pages;
  if (!pages) return null;
  for (const page of Object.values(pages)) {
    const source = (page as { thumbnail?: { source?: unknown } })?.thumbnail?.source;
    if (typeof source === "string") return source;
  }
  return null;
}

const cache = new Map<string, string | null>();

export async function fetchWikipediaPhoto(name: string): Promise<string | null> {
  if (cache.has(name)) return cache.get(name) ?? null;

  const params = new URLSearchParams({
    action: "query",
    generator: "search",
    gsrsearch: name,
    gsrlimit: "1",
    prop: "pageimages",
    format: "json",
    pithumbsize: String(THUMB_WIDTH),
    origin: "*",
  });

  let url: string | null = null;
  try {
    const res = await fetch(`https://en.wikipedia.org/w/api.php?${params.toString()}`);
    if (res.ok) {
      url = photoUrlFromApiResponse(await res.json());
    }
  } catch {
    url = null;
  }

  cache.set(name, url);
  return url;
}
