import type Sigma from "sigma";
import type { Relation, GraphData, ScoreScale } from "./types";

const FOCUS_RATIO = 0.15;

/** Centres the camera on a node. */
export function flyToNode(
  renderer: Sigma,
  nodeId: string,
  focus?: { x: number; y: number },
  duration = 500
): void {
  // Sigma's camera lives in an internally normalized ~[0,1] space (computed from the
  // graph's bounding box), NOT the raw graph.json coordinates (our [0,10000] canvas).
  const display = renderer.getNodeDisplayData(nodeId);
  if (!display) throw new Error(`no display data for node ${nodeId} (not rendered yet?)`);

  const camera = renderer.getCamera();
  let { x, y } = display;

  if (focus) {
    // viewportToFramedGraph, NOT viewportToGraph.
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

/** 16091 -> "16,000". */
function roughly(n: number): string {
  const step = 10 ** Math.max(0, Math.floor(Math.log10(n)) - 1);
  return (Math.round(n / step) * step).toLocaleString("en-US");
}

/** The sentence that stops a score being read as a school mark. */
export function rarityNote(weight: number, scale: ScoreScale): string {
  const bucket = Math.min(100, Math.max(0, Math.floor(weight * 100)));
  const above = Math.max(1, scale.tail[bucket]);
  return `1 in ${roughly(scale.pairs / above)} pairs score this high`
    + ` · a random pair scores 0%`
    + ` · the map's closest pair, ${formatSimilarity(scale.top)}`;
}

export function escapeHtml(value: string): string {
  return value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

/** How a similarity score reads in words. */
const RESEMBLANCE_FLOOR = 0.14;

export interface Resemblance {
  label: string;
  /** Drives the colour, so CSS never has to match on the wording. */
  slug: "strong" | "medium" | "faint";
}

const BANDS: { min: number; band: Resemblance }[] = [
  { min: 0.285, band: { label: "Lookalike", slug: "strong" } },
  { min: 0.22, band: { label: "Looks somewhat alike", slug: "medium" } },
  { min: RESEMBLANCE_FLOOR, band: { label: "Far resemblance", slug: "faint" } },
];

/** The band a score falls in, or null when it is below the floor. */
export function resemblanceBand(weight: number): Resemblance | null {
  return BANDS.find((b) => weight >= b.min)?.band ?? null;
}

/** The ranked list split into labelled groups, weakest group last. */
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
  /** One row of the similar-people list. */
  similar: SidebarRow[];
}

export interface SidebarRow {
  id: number;
  name: string;
  percent: string;
  weight: number;
  myPhoto: number;
  theirPhoto: number;
  /** What this person is to the one being viewed, when they are family. */
  relation?: Relation;
}

export function getSidebarData(data: GraphData, nodeId: number): SidebarData {
  const node = data.nodes.find((n) => n.id === nodeId);
  if (!node) throw new Error(`node ${nodeId} not found`);
  const nodesById = new Map(data.nodes.map((n) => [n.id, n]));
  const ranked = data.similar[String(nodeId)] ?? [];
  return {
    id: node.id,
    name: node.name,
    similar: ranked.map(([id, weight, myPhoto, theirPhoto, relation]) => ({
      id,
      name: nodesById.get(id)?.name ?? "Unknown",
      percent: formatSimilarity(weight),
      weight,
      myPhoto: myPhoto ?? 0,
      theirPhoto: theirPhoto ?? 0,
      ...(relation ? { relation } : {}),
    })),
  };
}
