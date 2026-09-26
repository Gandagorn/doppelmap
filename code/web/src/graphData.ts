import Graph from "graphology";
import { edgeColorForStrength, nodeColor } from "./theme";
import type { GraphData, GraphEdge } from "./types";

/** Maps a raw edge weight onto 0..1 within this dataset's own spread. */
function edgeLengthFade(length: number, medianLength: number): number {
  if (medianLength <= 0) return 1;
  // Starts at 1.5x the median rather than 3x.
  return Math.min(1, (medianLength * 1.5) / length);
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

/** Give one node the face that has just been cut for it. */
export function setFaceImage(graph: Graph, id: string, url: string): void {
  if (graph.hasNode(id)) graph.setNodeAttribute(id, "image", url);
}

export function buildGraphology(
  data: GraphData, faces = false
): Graph {
  const graph = new Graph({ type: "undirected", multi: false });
  for (const node of data.nodes) {
    graph.addNode(String(node.id), {
      label: node.name,
      x: node.x,
      y: node.y,
      // A face has to be big enough to be a face.
      size: faces
        ? 5 + Math.sqrt(node.deg) * 1.6
        : 1.5 + Math.sqrt(node.deg) * 0.6,
      color: nodeColor(),
      deg: node.deg,
    });
  }
  // Both width and opacity track strength.
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
      // Also retuned for face-sized nodes: thinner at the bottom so the weak majority
      // recedes, and a wider spread so the few strong links are the ones the eye
      // follows.
      size: 0.35 + strength * 1.5,
      color: edgeColorForStrength(strength, fade),
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
