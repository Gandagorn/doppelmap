import type Sigma from "sigma";
import type { GraphData } from "./types";

const FOCUS_RATIO = 0.15;

/** Centres the camera on a node.
 *
 *  `focus` is where on screen (in viewport pixels) the node should end up.
 *  Pass it when part of the canvas is covered by UI -- the mobile sidebar
 *  is a bottom sheet over half the screen, so a node centred in the
 *  viewport lands underneath it. Omit it to centre in the viewport. */
export function flyToNode(
  renderer: Sigma,
  nodeId: string,
  focus?: { x: number; y: number },
  duration = 500
): void {
  // Sigma's camera lives in an internally normalized ~[0,1] space (computed
  // from the graph's bounding box), NOT the raw graph.json coordinates
  // (our [0,10000] canvas). getNodeDisplayData returns the already-
  // normalized, cached position -- reading graph.getNodeAttribute directly
  // sends the camera thousands of units outside the graph's visible area.
  const display = renderer.getNodeDisplayData(nodeId);
  if (!display) throw new Error(`no display data for node ${nodeId} (not rendered yet?)`);

  const camera = renderer.getCamera();
  let { x, y } = display;

  if (focus) {
    // Converting at the *target* camera state, not the current one: the
    // graph-space size of a pixel depends on the zoom we're flying to, so
    // measuring at the current zoom would offset by the wrong amount
    // whenever this changes the zoom level (which it usually does).
    const targetState = { ...camera.getState(), x: display.x, y: display.y, ratio: FOCUS_RATIO };
    const { width, height } = renderer.getDimensions();
    const atCentre = renderer.viewportToGraph(
      { x: width / 2, y: height / 2 },
      { cameraState: targetState }
    );
    const atFocus = renderer.viewportToGraph(focus, { cameraState: targetState });
    // Shifting the camera the opposite way moves the node toward `focus`.
    x -= atFocus.x - atCentre.x;
    y -= atFocus.y - atCentre.y;
  }

  camera.animate({ x, y, ratio: FOCUS_RATIO }, { duration });
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
  // `weight` is the raw similarity alongside the display-formatted
  // `percent` -- the pair-comparison view re-formats it itself.
  similar: { id: number; name: string; percent: string; weight: number; thumb: string }[];
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
      weight,
      thumb: nodesById.get(id)?.thumb ?? "",
    })),
  };
}
