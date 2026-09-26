export interface GraphNode {
  id: number;
  name: string;
  /** Wikidata Q-number, which is what a vote is stored against. */
  qid?: string;
  x: number;
  y: number;
  deg: number;
  /** How many Commons photographs were accepted as this person. */
  faces: number;
}

export type GraphEdge = [number, number, number];

/** `[otherId, similarity, myPhotoIndex, theirPhotoIndex, related?]`. */
export type Relation = "parent" | "child" | "sibling" | "spouse" | "family";
export type SimilarEntry =
  | [number, number, number, number]
  | [number, number, number, number, Relation];

/** Where a score sits among every pair the map could have formed. */
export interface ScoreScale {
  /** Every unordered pair of people on the map, scored or not. */
  pairs: number;
  /** tail[i] is how many of those score at least i/100. */
  tail: number[];
  /** The highest score any pair reaches -- the top of the usable scale. */
  top: number;
}

export interface GraphData {
  meta: { version: string; count: number; k: number; scale: ScoreScale };
  nodes: GraphNode[];
  edges: GraphEdge[];
  similar: Record<string, SimilarEntry[]>;
  /** Node ids worth landing a first-time visitor on: recognisable, with a match strong enough to see, and not a relative.
   */
  landing?: number[];
}

/** One accepted face: which Commons file, and where the face sits in it. */
export interface PhotoRef {
  /** Commons filename, without the "File:" prefix. */
  f: string;
  /** Face box as fractions of the image: [x1, y1, x2, y2]. */
  b: [number, number, number, number];
  /** The image's own aspect ratio (width / height). */
  a: number;
  /** Credit and licence, already reduced to plain text. */
  c: string;
  l: string;
  /** A thumbnail URL the browser may read pixels from, on the first photo of each person only: the map cuts its face out of this one.
   */
  u?: string;
}

/** Keyed by node id as a string, best photo first. */
export type PhotoData = Record<string, PhotoRef[]>;
