// Live, on-demand lookup of a real profile photo (and the Wikipedia page it
// came from) -- called when a node's sidebar opens, not baked into
// graph.json at dataset-build time, so dataset size never bounds how long
// generation takes. Wikipedia's API supports direct browser calls via
// `origin=*` (CORS), so this needs no server of our own.
const THUMB_WIDTH = 192; // 2x the 96px display size, for retina

export interface WikipediaInfo {
  photoUrl: string | null;
  pageUrl: string | null;
}

const NO_INFO: WikipediaInfo = { photoUrl: null, pageUrl: null };

export function wikipediaInfoFromApiResponse(data: unknown): WikipediaInfo {
  const pages = (data as { query?: { pages?: Record<string, unknown> } } | null | undefined)
    ?.query?.pages;
  if (!pages) return NO_INFO;
  for (const page of Object.values(pages)) {
    const p = page as { thumbnail?: { source?: unknown }; fullurl?: unknown };
    const source = p?.thumbnail?.source;
    const fullurl = p?.fullurl;
    return {
      photoUrl: typeof source === "string" ? source : null,
      pageUrl: typeof fullurl === "string" ? fullurl : null,
    };
  }
  return NO_INFO;
}

// Cache the in-flight promise, not just the resolved value -- preloading a
// person's neighbors and then hovering/clicking one moments later must not
// fire a second request for the same name while the first is still pending.
const cache = new Map<string, Promise<WikipediaInfo>>();

async function fetchFromApi(name: string): Promise<WikipediaInfo> {
  const params = new URLSearchParams({
    action: "query",
    generator: "search",
    gsrsearch: name,
    gsrlimit: "1",
    prop: "pageimages|info",
    inprop: "url",
    format: "json",
    pithumbsize: String(THUMB_WIDTH),
    origin: "*",
  });

  try {
    const res = await fetch(`https://en.wikipedia.org/w/api.php?${params.toString()}`);
    if (!res.ok) return NO_INFO;
    return wikipediaInfoFromApiResponse(await res.json());
  } catch {
    return NO_INFO;
  }
}

export function fetchWikipediaInfo(name: string): Promise<WikipediaInfo> {
  let pending = cache.get(name);
  if (!pending) {
    pending = fetchFromApi(name);
    cache.set(name, pending);
  }
  return pending;
}
