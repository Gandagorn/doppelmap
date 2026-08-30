import Sigma from "sigma";
import { loadGraphData, buildGraphology } from "./graphData";
import { getDisplayMode } from "./sigmaSetup";
import { searchNames } from "./search";
import { flyToNode } from "./interactions";

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

  const searchInput = document.getElementById("search") as HTMLInputElement;
  const resultsEl = document.getElementById("search-results") as HTMLDivElement;

  let debounceHandle: ReturnType<typeof setTimeout> | undefined;
  searchInput.addEventListener("input", () => {
    clearTimeout(debounceHandle);
    debounceHandle = setTimeout(() => {
      const matches = searchNames(data.nodes, searchInput.value);
      resultsEl.innerHTML = "";
      for (const node of matches) {
        const item = document.createElement("div");
        item.className = "search-result";
        item.textContent = node.name;
        item.addEventListener("click", () => {
          flyToNode(renderer, graph, String(node.id));
          resultsEl.innerHTML = "";
          searchInput.value = node.name;
        });
        resultsEl.appendChild(item);
      }
    }, 100);
  });
}

bootstrap().catch((err) => {
  console.error(err);
});
