import type Sigma from "sigma";
import type Graph from "graphology";
import type { GraphData } from "./types";

export function flyToNode(
  renderer: Sigma,
  graph: Graph,
  nodeId: string,
  duration = 500
): void {
  const x = graph.getNodeAttribute(nodeId, "x") as number;
  const y = graph.getNodeAttribute(nodeId, "y") as number;
  renderer.getCamera().animate({ x, y, ratio: 0.15 }, { duration });
}

export function formatSimilarity(weight: number): string {
  return `${Math.round(weight * 1000) / 10}%`;
}

export interface SidebarData {
  id: number;
  name: string;
  thumb: string;
  attr: string;
  similar: { id: number; name: string; percent: string }[];
}

export function getSidebarData(data: GraphData, nodeId: number): SidebarData {
  const node = data.nodes.find((n) => n.id === nodeId);
  if (!node) throw new Error(`node ${nodeId} not found`);
  const nodesById = new Map(data.nodes.map((n) => [n.id, n]));
  const ranked = data.similar[String(nodeId)] ?? [];
  return {
    id: node.id,
    name: node.name,
    thumb: node.thumb,
    attr: node.attr,
    similar: ranked.map(([id, weight]) => ({
      id,
      name: nodesById.get(id)?.name ?? "Unknown",
      percent: formatSimilarity(weight),
    })),
  };
}
