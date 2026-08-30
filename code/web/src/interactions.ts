import type Sigma from "sigma";
import type Graph from "graphology";

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
