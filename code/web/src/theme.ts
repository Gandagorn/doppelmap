// The map's palette. One dark theme, tuned against the aurora background the
// page paints behind Sigma's transparent canvases.

/** Node fill: warm clay, close enough to skin to sit under a face. */
const ACCENT = "#e0a882";

/** Hover-dimmed nodes: present, but clearly out of the way. */
export const DIM_NODE_COLOR = "#6d7893";

/** The darkest point of the background, so edges can be mixed against it. */
const MAP_BG = [62, 72, 102] as const;

/** Mix `hex` into the background: t=0 invisible, t=1 full colour. */
function mixIntoBackground(hex: string, t: number): string {
  const f = Math.min(1, Math.max(0, t));
  const ch = (i: number) =>
    Math.round(MAP_BG[i] + (parseInt(hex.slice(1 + i * 2, 3 + i * 2), 16) - MAP_BG[i]) * f);
  return `rgb(${ch(0)}, ${ch(1)}, ${ch(2)})`;
}

const FADED_EDGE_MIX = 0.16;
const FADED_EDGE_INK = "#b9c4dd";

/** Edges away from the highlighted node: enough to keep the map's shape. */
export function fadedEdgeColor(): string {
  return mixIntoBackground(FADED_EDGE_INK, FADED_EDGE_MIX);
}

/** The selected node: warm amber, far from both the accent and the dim grey. */
export const SELECTED_NODE_COLOR = "#ffcf7a";

export function nodeColor(): string {
  return ACCENT;
}

export function hexToRgba(hex: string, alpha: number): string {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

export function edgeColor(alpha = 0.35): string {
  return hexToRgba(ACCENT, alpha);
}

const EDGE_MIX_MIN = 0.22;
const EDGE_MIX_MAX = 0.72;

/** Strong links glow, weak ones recede. `strength` is a 0..1 position within
 *  the dataset's own weight spread, not a raw similarity. */
export function edgeColorForStrength(strength: number, fade = 1): string {
  const t = Math.min(1, Math.max(0, strength));
  const f = Math.min(1, Math.max(0, fade));
  // Strong links lean teal: the complement of skin, so they read against faces.
  const ink = t > 0.72 ? "#ffc27a" : ACCENT;
  return mixIntoBackground(ink, (EDGE_MIX_MIN + (EDGE_MIX_MAX - EDGE_MIX_MIN) * t) * f);
}

/** Sigma paints labels on its own canvas, so they need an explicit ink. */
export function labelColor(): string {
  return "#f2f6ff";
}
