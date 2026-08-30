import type Sigma from "sigma";
import type { GraphData } from "./types";

export function flyToNode(renderer: Sigma, nodeId: string, duration = 500): void {
  // Sigma's camera lives in an internally normalized ~[0,1] space (computed
  // from the graph's bounding box), NOT the raw graph.json coordinates
  // (our [0,10000] canvas). getNodeDisplayData returns the already-
  // normalized, cached position -- reading graph.getNodeAttribute directly
  // sends the camera thousands of units outside the graph's visible area.
  const display = renderer.getNodeDisplayData(nodeId);
  if (!display) throw new Error(`no display data for node ${nodeId} (not rendered yet?)`);
  renderer.getCamera().animate({ x: display.x, y: display.y, ratio: 0.15 }, { duration });
}

export function formatSimilarity(weight: number): string {
  return `${Math.round(weight * 1000) / 10}%`;
}

export function escapeHtml(value: string): string {
  return value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

export interface SidebarData {
  id: number;
  name: string;
  thumb: string;
  attr: string;
  similar: { id: number; name: string; percent: string; thumb: string }[];
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
      thumb: nodesById.get(id)?.thumb ?? "",
    })),
  };
}
