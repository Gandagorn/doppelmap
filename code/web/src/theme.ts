// Single accent color for graph rendering -- there's no meaningful category
// to color-encode (clusters in the synthetic dataset are arbitrary), so one
// clean accent beats an arbitrary rainbow. Values are the dataviz palette's
// categorical slot 1 (blue), light and dark variants.
const ACCENT = { light: "#2a78d6", dark: "#3987e5" } as const;

// Hover-dimmed node color: a fixed muted gray that reads against both the
// light and dark background gradients, so it doesn't need a theme split.
export const DIM_NODE_COLOR = "#9ca3af";

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
