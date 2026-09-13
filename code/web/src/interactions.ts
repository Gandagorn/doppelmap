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
    // viewportToFramedGraph, NOT viewportToGraph. Same trap as the display
    // coordinates above: viewportToGraph runs the result back through
    // normalizationFunction.inverse and hands back raw graph.json units
    // (our [0,10000] canvas), which are meaningless as a camera offset --
    // subtracting one from a ~0.5 camera coordinate throws the view
    // thousands of units off and the graph disappears entirely. The
    // "framed" space is the normalized one the camera actually lives in.
    //
    // Converting at the *target* camera state, not the current one: the
    // size of a pixel in that space depends on the zoom we're flying to,
    // so measuring at the current zoom would offset by the wrong amount
    // whenever this changes the zoom level (which it usually does).
    const targetState = { ...camera.getState(), x: display.x, y: display.y, ratio: FOCUS_RATIO };
    const { width, height } = renderer.getDimensions();
    const atCentre = renderer.viewportToFramedGraph(
      { x: width / 2, y: height / 2 },
      { cameraState: targetState }
    );
    const atFocus = renderer.viewportToFramedGraph(focus, { cameraState: targetState });
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

/** How a similarity score reads in words.
 *
 *  A raw cosine means nothing to a visitor, and the numbers are packed into
 *  a narrow band, so the difference between 15% and 17% is noise dressed up
 *  as precision. Over the 14,430 ranked pairs in the full collection these
 *  cuts leave 0.6% Lookalike, 5.7% somewhat alike and 70% a far resemblance,
 *  hiding the bottom 23.5% -- which is low enough that nobody is left with
 *  an empty list.
 */
export const RESEMBLANCE_FLOOR = 0.14;

export interface Resemblance {
  label: string;
  /** Drives the colour, so CSS never has to match on the wording. */
  slug: "strong" | "medium" | "faint";
}

const BANDS: { min: number; band: Resemblance }[] = [
  { min: 0.3, band: { label: "Lookalike", slug: "strong" } },
  { min: 0.22, band: { label: "Looks somewhat alike", slug: "medium" } },
  { min: RESEMBLANCE_FLOOR, band: { label: "Far resemblance", slug: "faint" } },
];

/** The band a score falls in, or null when it is below the floor. */
export function resemblanceBand(weight: number): Resemblance | null {
  return BANDS.find((b) => weight >= b.min)?.band ?? null;
}

/** The ranked list split into labelled groups, weakest group last.
 *
 *  Entries below the floor are dropped entirely. No one in the current
 *  collection loses their whole list that way, but a thinner dataset could,
 *  so callers still have to handle an empty result. */
export function groupByResemblance<T extends { weight: number }>(
  entries: T[]
): { band: Resemblance; entries: T[] }[] {
  const groups: { band: Resemblance; entries: T[] }[] = [];
  for (const entry of entries) {
    const band = resemblanceBand(entry.weight);
    if (!band) continue;
    const last = groups[groups.length - 1];
    if (last && last.band.slug === band.slug) last.entries.push(entry);
    else groups.push({ band, entries: [entry] });
  }
  return groups;
}

export interface SidebarData {
  id: number;
  name: string;
  /** One row of the similar-people list.
   *
   *  `weight` is the raw similarity alongside the display-formatted
   *  `percent`, since the comparison view re-formats it at a larger size.
   *  `myPhoto` / `theirPhoto` index into each person's photos.json entry
   *  and name the two photographs that actually match. */
  similar: SidebarRow[];
}

export interface SidebarRow {
  id: number;
  name: string;
  percent: string;
  weight: number;
  myPhoto: number;
  theirPhoto: number;
}

export function getSidebarData(data: GraphData, nodeId: number): SidebarData {
  const node = data.nodes.find((n) => n.id === nodeId);
  if (!node) throw new Error(`node ${nodeId} not found`);
  const nodesById = new Map(data.nodes.map((n) => [n.id, n]));
  const ranked = data.similar[String(nodeId)] ?? [];
  return {
    id: node.id,
    name: node.name,
    similar: ranked.map(([id, weight, myPhoto, theirPhoto]) => ({
      id,
      name: nodesById.get(id)?.name ?? "Unknown",
      percent: formatSimilarity(weight),
      weight,
      myPhoto: myPhoto ?? 0,
      theirPhoto: theirPhoto ?? 0,
    })),
  };
}
