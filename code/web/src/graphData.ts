import Graph from "graphology";
import { edgeColor, nodeColor } from "./theme";
import type { GraphData } from "./types";

export function buildGraphology(data: GraphData, isDark: boolean): Graph {
  const graph = new Graph({ type: "undirected", multi: false });
  for (const node of data.nodes) {
    graph.addNode(String(node.id), {
      label: node.name,
      x: node.x,
      y: node.y,
      size: 3 + Math.sqrt(node.deg),
      color: nodeColor(isDark),
      thumb: node.thumb,
      attr: node.attr,
      deg: node.deg,
    });
  }
  for (const [a, b, weight] of data.edges) {
    graph.addEdge(String(a), String(b), { weight, size: 0.5 + weight, color: edgeColor(isDark) });
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
