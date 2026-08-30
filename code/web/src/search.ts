import type { GraphNode } from "./types";

function normalize(value: string): string {
  return value
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .toLowerCase();
}

export function searchNames(nodes: GraphNode[], query: string, limit = 8): GraphNode[] {
  const q = normalize(query.trim());
  if (!q) return [];

  const prefixMatches: GraphNode[] = [];
  const substringMatches: GraphNode[] = [];
  for (const node of nodes) {
    const normalizedName = normalize(node.name);
    if (normalizedName.startsWith(q)) {
      prefixMatches.push(node);
    } else if (normalizedName.includes(q)) {
      substringMatches.push(node);
    }
  }
  return [...prefixMatches, ...substringMatches].slice(0, limit);
}
