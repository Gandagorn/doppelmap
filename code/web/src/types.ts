export interface GraphNode {
  id: number;
  name: string;
  x: number;
  y: number;
  deg: number;
  /** How many Commons photographs were accepted as this person. */
  faces: number;
}

export type GraphEdge = [number, number, number];

/** `[otherId, similarity, myPhotoIndex, theirPhotoIndex]`.
 *
 *  The two indices point into each person's entry in photos.json and name
 *  the pair of photographs that actually match, worked out at build time
 *  so the browser never needs the embeddings. */
export type SimilarEntry = [number, number, number, number];

export interface GraphData {
  meta: { version: string; count: number; k: number };
  nodes: GraphNode[];
  edges: GraphEdge[];
  similar: Record<string, SimilarEntry[]>;
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
}

/** Keyed by node id as a string, best photo first. */
export type PhotoData = Record<string, PhotoRef[]>;
