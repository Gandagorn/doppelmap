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

async function queryWikipedia(extra: Record<string, string>): Promise<WikipediaInfo> {
  const params = new URLSearchParams({
    action: "query",
    prop: "pageimages|info",
    inprop: "url",
    format: "json",
    pithumbsize: String(THUMB_WIDTH),
    origin: "*",
    ...extra,
  });
  const res = await fetch(`https://en.wikipedia.org/w/api.php?${params.toString()}`);
  if (!res.ok) return NO_INFO;
  return wikipediaInfoFromApiResponse(await res.json());
}

async function fetchFromApi(name: string): Promise<WikipediaInfo> {
  try {
    // Try the name as an exact article title first -- our names are
    // themselves sourced from Wikipedia, so this usually lands on the
    // right article directly. `generator=search` ranks by relevance and
    // can surface a different, more "notable" same-named person instead
    // (e.g. a common name matching a more-searched-for unrelated page).
    const exact = await queryWikipedia({ titles: name, redirects: "1" });
    if (exact.pageUrl) return exact;
    return await queryWikipedia({ generator: "search", gsrsearch: name, gsrlimit: "1" });
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
