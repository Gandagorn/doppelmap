import { createClient, type SupabaseClient } from "@supabase/supabase-js";

// The publishable key is meant to be public -- it identifies the project, it does not
// authorise anything.
const SUPABASE_URL = "https://ywanrluqktgzldtfnwuo.supabase.co";
const SUPABASE_KEY = "sb_publishable_D16wo7a0pklM37EIGiT2Sg_-XFiZh-g";

export interface PairTally {
  yes: number;
  no: number;
  total: number;
}

let client: SupabaseClient | null = null;

function supabase(): SupabaseClient {
  if (!client) {
    client = createClient(SUPABASE_URL, SUPABASE_KEY, {
      auth: { persistSession: true, autoRefreshToken: true },
    });
  }
  return client;
}

let signingIn: Promise<boolean> | null = null;

/** Signs the visitor in anonymously, once, and remembers it. */
async function ensureSignedIn(): Promise<boolean> {
  const { data } = await supabase().auth.getSession();
  if (data.session) return true;
  if (!signingIn) {
    signingIn = supabase().auth.signInAnonymously()
      .then(({ error }) => !error)
      .catch(() => false)
      .finally(() => { signingIn = null; });
  }
  return signingIn;
}

/** Wikidata ids, in the order the table stores them. */
export function orderPair(a: string, b: string): [string, string] {
  return a < b ? [a, b] : [b, a];
}

const tallies = new Map<string, PairTally>();

function cacheKey(a: string, b: string): string {
  return orderPair(a, b).join("|");
}

/** Everyone's answer for a pair, or null when the service cannot be reached. */
export async function pairTally(a: string, b: string): Promise<PairTally | null> {
  const key = cacheKey(a, b);
  const cached = tallies.get(key);
  if (cached) return cached;
  const [personA, personB] = orderPair(a, b);
  try {
    const { data, error } = await supabase()
      .rpc("get_pair_votes", { p_person_a: personA, p_person_b: personB });
    if (error) return null;
    const row = (Array.isArray(data) ? data[0] : data) as {
      yes_votes?: number; no_votes?: number; total_votes?: number;
    } | undefined;
    const tally: PairTally = {
      yes: Number(row?.yes_votes ?? 0),
      no: Number(row?.no_votes ?? 0),
      total: Number(row?.total_votes ?? 0),
    };
    tallies.set(key, tally);
    return tally;
  } catch {
    return null;
  }
}

/** Records a vote and returns the new totals. */
export async function castVote(
  a: string, b: string, looksAlike: boolean
): Promise<PairTally | null> {
  if (!(await ensureSignedIn())) return null;
  const [personA, personB] = orderPair(a, b);
  try {
    const { error } = await supabase().rpc("cast_pair_vote", {
      p_person_a: personA, p_person_b: personB, p_looks_alike: looksAlike,
    });
    if (error) return null;
  } catch {
    return null;
  }
  tallies.delete(cacheKey(a, b));
  return pairTally(a, b);
}

/** Which way this visitor voted on a pair, for the life of the tab. */
const myVotes = new Map<string, boolean>();

export function rememberVote(a: string, b: string, looksAlike: boolean): void {
  myVotes.set(cacheKey(a, b), looksAlike);
}

export function myVote(a: string, b: string): boolean | undefined {
  return myVotes.get(cacheKey(a, b));
}

/** "68% of 25 people say yes", or the honest version when nobody has voted. */
export function describeTally(tally: PairTally): string {
  if (!tally.total) return "No votes yet — be the first.";
  const share = Math.round((tally.yes / tally.total) * 100);
  const people = tally.total === 1 ? "1 person" : `${tally.total} people`;
  return `${share}% of ${people} say yes`;
}
