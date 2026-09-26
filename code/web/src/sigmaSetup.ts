export type DisplayMode = "dot" | "label";

const LABEL_THRESHOLD_RATIO = 0.5;

export function getDisplayMode(cameraRatio: number): DisplayMode {
  return cameraRatio <= LABEL_THRESHOLD_RATIO ? "label" : "dot";
}
