// Maps a 0-100 slider percentile to a popularity threshold: nodes with
// popularity >= threshold stay visible. 0 = show everyone, 100 = show only
// the single most popular node.
export function percentileThreshold(values: number[], percentile: number): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const clamped = Math.min(100, Math.max(0, percentile));
  const index = Math.floor((clamped / 100) * (sorted.length - 1));
  return sorted[index];
}
