import Graph from "graphology";
import type { GraphData } from "./types";

export function buildGraphology(data: GraphData): Graph {
  const graph = new Graph({ type: "undirected", multi: false });
  for (const node of data.nodes) {
    graph.addNode(String(node.id), {
      label: node.name,
      x: node.x,
      y: node.y,
      size: 3 + Math.sqrt(node.deg),
      thumb: node.thumb,
      attr: node.attr,
      deg: node.deg,
    });
  }
  for (const [a, b, weight] of data.edges) {
    graph.addEdge(String(a), String(b), { weight, size: 0.5 + weight });
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
