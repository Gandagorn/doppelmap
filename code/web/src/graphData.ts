import Graph from "graphology";
import { edgeColorForStrength, nodeColor } from "./theme";
import type { GraphData, GraphEdge } from "./types";

/** Maps a raw edge weight onto 0..1 within this dataset's own spread.
 *
 *  Real weights occupy a narrow, level-dependent band (top20 runs
 *  0.154-0.453 with a p25-p75 of just 0.220-0.266), so any fixed scale
 *  either flattens them or clips them. Bounds are the 5th/95th percentile
 *  rather than min/max so one unusually strong pair can't compress
 *  everything else into the bottom of the range. */
/** How much to fade an edge whose endpoints ended up far apart on screen.
 *
 *  A mutual-kNN edge says two people are alike in 512-d space, but UMAP
 *  can't honour every such relationship in 2-D -- some connected pairs
 *  land on opposite sides of the map, and drawing those as full-strength
 *  straight lines produces long streaks right across the view that read
 *  as noise and hide the local structure. Fading them in proportion to
 *  how far past typical they stretch is the honest version: position
 *  genuinely doesn't represent those pairs well.
 *
 *  Edges up to 3x the median length are untouched; beyond that the fade
 *  falls off as 1/length. */
export function edgeLengthFade(length: number, medianLength: number): number {
  if (medianLength <= 0) return 1;
  return Math.min(1, (medianLength * 3) / length);
}

export function edgeStrengthScale(edges: GraphEdge[]): (weight: number) => number {
  if (edges.length === 0) return () => 1;
  const sorted = edges.map((e) => e[2]).sort((a, b) => a - b);
  const at = (p: number) => sorted[Math.round(p * (sorted.length - 1))];
  const lo = at(0.05);
  const hi = at(0.95);
  if (hi <= lo) return () => 1;
  return (weight) => Math.min(1, Math.max(0, (weight - lo) / (hi - lo)));
}

export function buildGraphology(data: GraphData, isDark: boolean): Graph {
  const graph = new Graph({ type: "undirected", multi: false });
  for (const node of data.nodes) {
    graph.addNode(String(node.id), {
      label: node.name,
      x: node.x,
      y: node.y,
      size: 1.5 + Math.sqrt(node.deg) * 0.6,
      color: nodeColor(isDark),
      deg: node.deg,
    });
  }
  // Both width and opacity track strength. The old `0.5 + weight` gave
  // every edge in the dataset a width between 0.65 and 0.95 -- a 0.3px
  // spread, which is no spread at all -- with one flat color on top, so
  // the map couldn't show which links were the interesting ones.
  const strengthOf = edgeStrengthScale(data.edges);
  const nodeById = new Map(data.nodes.map((n) => [n.id, n]));
  const lengthOf = ([a, b]: GraphEdge) => {
    const na = nodeById.get(a);
    const nb = nodeById.get(b);
    return na && nb ? Math.hypot(na.x - nb.x, na.y - nb.y) : 0;
  };
  const sortedLengths = data.edges.map(lengthOf).sort((x, y) => x - y);
  const medianLength = sortedLengths.length
    ? sortedLengths[Math.floor(sortedLengths.length / 2)]
    : 0;

  for (const edge of data.edges) {
    const [a, b, weight] = edge;
    const strength = strengthOf(weight);
    const fade = edgeLengthFade(lengthOf(edge), medianLength);
    graph.addEdge(String(a), String(b), {
      weight,
      size: 0.4 + strength * 1.7,
      color: edgeColorForStrength(isDark, strength, fade),
    });
  }
  return graph;
}

export async function loadGraphData(url: string): Promise<GraphData> {
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`failed to load graph data: ${res.status} ${res.statusText}`);
  }
  return (await res.json()) as GraphData;
}
