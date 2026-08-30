export interface GraphNode {
  id: number;
  name: string;
  x: number;
  y: number;
  deg: number;
  thumb: string;
  attr: string;
}

export type GraphEdge = [number, number, number];

export interface GraphData {
  meta: { version: string; count: number; k: number };
  nodes: GraphNode[];
  edges: GraphEdge[];
  similar: Record<string, [number, number][]>;
}
