// Single accent color for graph rendering -- there's no meaningful category
// to color-encode (clusters in the synthetic dataset are arbitrary), so one
// clean accent beats an arbitrary rainbow. Values are the dataviz palette's
// categorical slot 1 (blue), light and dark variants.
const ACCENT = { light: "#2a78d6", dark: "#3987e5" } as const;

// Hover-dimmed node color: a fixed muted gray that reads against both the
// light and dark background gradients, so it doesn't need a theme split.
export const DIM_NODE_COLOR = "#9ca3af";

// The map background each theme actually paints, so edge colours can be
// mixed against it here instead of relying on alpha at draw time.
const MAP_BG = { light: [246, 249, 252], dark: [10, 14, 22] } as const;

/** Mix `hex` into the theme background: t=0 is invisible, t=1 is full colour.
 *
 *  Returns an opaque colour on purpose. Sigma draws edges in WebGL against a
 *  transparent canvas, so a translucent edge does not composite against the
 *  CSS background the way its rgba() suggests, and worse, ~2,100 edges
 *  overlapping build their alpha up until a pixel reaches the base colour.
 *  That accumulation was the whole problem: identical code rendered 1.2% ink
 *  coverage in light mode and 63.7% in dark. Opaque colours cannot stack, so
 *  a dense region looks the same as a single line.
 */
function mixIntoBackground(hex: string, isDark: boolean, t: number): string {
  const bg = isDark ? MAP_BG.dark : MAP_BG.light;
  const f = Math.min(1, Math.max(0, t));
  const ch = (i: number) =>
    Math.round(bg[i] + (parseInt(hex.slice(1 + i * 2, 3 + i * 2), 16) - bg[i]) * f);
  return `rgb(${ch(0)}, ${ch(1)}, ${ch(2)})`;
}

// Edges not touching the highlighted node: present enough to keep the map's
// shape, quiet enough not to compete with the selection. Something is always
// selected, so this is what almost every edge is drawn in.
const FADED_EDGE_MIX = { light: 0.22, dark: 0.34 };
const FADED_EDGE_INK = "#475569";

export function fadedEdgeColor(isDark: boolean): string {
  return mixIntoBackground(
    FADED_EDGE_INK, isDark, isDark ? FADED_EDGE_MIX.dark : FADED_EDGE_MIX.light
  );
}

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

// How faint the weakest drawn edge is vs. the strongest, as a mix into the
// background rather than an alpha.
const EDGE_MIX_MIN = 0.12;
const EDGE_MIX_MAX = 0.75;

/** `strength` is a 0..1 position within the dataset's own weight spread,
 *  NOT a raw similarity -- see edgeStrengthScale in graphData.ts. Raw
 *  weights all sit inside a narrow band (the middle half of the real data
 *  spans just ~0.22 to ~0.29), so feeding them in directly would paint
 *  every edge essentially the same. */
export function edgeColorForStrength(isDark: boolean, strength: number, fade = 1): string {
  const t = Math.min(1, Math.max(0, strength));
  const f = Math.min(1, Math.max(0, fade));
  return mixIntoBackground(
    nodeColor(isDark), isDark, (EDGE_MIX_MIN + (EDGE_MIX_MAX - EDGE_MIX_MIN) * t) * f
  );
}

// Sigma paints labels on its own canvas and defaults to a dark ink, which on
// the dark theme's near-black background is unreadable -- the selected
// person's own name was the least legible thing on the map.
const LABEL_INK = { light: "#1a2233", dark: "#e2e8f0" } as const;

export function labelColor(isDark: boolean): string {
  return isDark ? LABEL_INK.dark : LABEL_INK.light;
}
