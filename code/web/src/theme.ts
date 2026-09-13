// Single accent color for graph rendering -- there's no meaningful category
// to color-encode (clusters in the synthetic dataset are arbitrary), so one
// clean accent beats an arbitrary rainbow. Values are the dataviz palette's
// categorical slot 1 (blue), light and dark variants.
const ACCENT = { light: "#2a78d6", dark: "#3987e5" } as const;

// Hover-dimmed node color: a fixed muted gray that reads against both the
// light and dark background gradients, so it doesn't need a theme split.
export const DIM_NODE_COLOR = "#9ca3af";

// Edges not touching the highlighted node: barely-there, so the map keeps
// its shape instead of going blank around the selection.
export const FADED_EDGE_COLOR = "rgba(148, 163, 184, 0.13)";

// The selected node's own color: a muted amber, far enough from the
// accent blue and the dim gray to read as "selected" without shouting.
// Same fixed-value reasoning as DIM_NODE_COLOR above.
export const SELECTED_NODE_COLOR = "#c99a5b";

export function nodeColor(isDark: boolean): string {
  return isDark ? ACCENT.dark : ACCENT.light;
}

export function hexToRgba(hex: string, alpha: number): string {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

export function edgeColor(isDark: boolean, alpha = 0.35): string {
  return hexToRgba(nodeColor(isDark), alpha);
}

// How faint the weakest drawn edge is vs. the strongest.
const EDGE_ALPHA_MIN = 0.07;
const EDGE_ALPHA_MAX = 0.8;

/** `strength` is a 0..1 position within the dataset's own weight spread,
 *  NOT a raw similarity -- see edgeStrengthScale in graphData.ts. Raw
 *  weights all sit inside a narrow band (the middle half of the real data
 *  spans just ~0.22 to ~0.29), so feeding them in directly would paint
 *  every edge essentially the same. */
export function edgeColorForStrength(isDark: boolean, strength: number, fade = 1): string {
  const t = Math.min(1, Math.max(0, strength));
  const f = Math.min(1, Math.max(0, fade));
  return hexToRgba(nodeColor(isDark), (EDGE_ALPHA_MIN + (EDGE_ALPHA_MAX - EDGE_ALPHA_MIN) * t) * f);
}
