import Sigma from "sigma";
import { loadGraphData, buildGraphology } from "./graphData";
import { getDisplayMode } from "./sigmaSetup";

async function bootstrap() {
  const data = await loadGraphData("/data/graph.json");
  const graph = buildGraphology(data);

  const container = document.getElementById("graph-container");
  if (!container) throw new Error("#graph-container not found in DOM");

  const renderer = new Sigma(graph, container, {
    labelRenderedSizeThreshold: 0,
  });

  renderer.setSetting("nodeReducer", (_node, attrs) => {
    const mode = getDisplayMode(renderer.getCamera().ratio);
    return mode === "dot" ? { ...attrs, label: "" } : attrs;
  });
}

bootstrap().catch((err) => {
  console.error(err);
});
