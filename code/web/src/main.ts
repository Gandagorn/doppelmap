import Sigma from "sigma";
import { loadGraphData, buildGraphology } from "./graphData";
import { getDisplayMode } from "./sigmaSetup";
import { searchNames } from "./search";
import { flyToNode, getSidebarData } from "./interactions";

async function bootstrap() {
  const data = await loadGraphData("/data/graph.json");
  const graph = buildGraphology(data);

  const container = document.getElementById("graph-container");
  if (!container) throw new Error("#graph-container not found in DOM");

  const renderer = new Sigma(graph, container, {
    labelRenderedSizeThreshold: 0,
  });

  // Total UI state: which node is selected (sidebar open) and which is
  // hovered (dims everything else). Just a plain object mutated in place —
  // two fields don't need a reducer.
  const selection: { selectedId: number | null; hoveredId: number | null } = {
    selectedId: null,
    hoveredId: null,
  };
  const sidebarEl = document.getElementById("sidebar") as HTMLElement;

  function renderSidebar() {
    if (selection.selectedId === null) {
      sidebarEl.hidden = true;
      sidebarEl.innerHTML = "";
      return;
    }
    const info = getSidebarData(data, selection.selectedId);
    sidebarEl.hidden = false;
    sidebarEl.innerHTML = `
      <img src="/data/${info.thumb}" width="96" height="96" alt="${info.name}" />
      <h2>${info.name}</h2>
      <p class="attr">${info.attr}</p>
      <ul class="similar-list">
        ${info.similar
          .map((s) => `<li data-id="${s.id}">${s.name} — ${s.percent}</li>`)
          .join("")}
      </ul>
    `;
    sidebarEl.querySelectorAll<HTMLLIElement>("li[data-id]").forEach((li) => {
      li.addEventListener("click", () => {
        selection.selectedId = Number(li.dataset.id);
        flyToNode(renderer, graph, String(selection.selectedId));
        renderSidebar();
      });
    });
  }

  renderer.on("clickNode", ({ node }) => {
    selection.selectedId = Number(node);
    flyToNode(renderer, graph, node);
    renderSidebar();
  });

  renderer.on("enterNode", ({ node }) => {
    selection.hoveredId = Number(node);
    renderer.refresh();
  });

  renderer.on("leaveNode", () => {
    selection.hoveredId = null;
    renderer.refresh();
  });

  renderer.on("clickStage", () => {
    selection.selectedId = null;
    renderSidebar();
  });

  window.addEventListener("keydown", (evt) => {
    if (evt.key === "Escape") {
      selection.selectedId = null;
      renderSidebar();
    }
  });

  renderer.setSetting("nodeReducer", (nodeId, attrs) => {
    const mode = getDisplayMode(renderer.getCamera().ratio);
    const display = { ...attrs };
    if (mode === "dot") display.label = "";

    if (selection.hoveredId !== null) {
      const hoveredKey = String(selection.hoveredId);
      const isHovered = nodeId === hoveredKey;
      const isNeighbor = graph.areNeighbors(nodeId, hoveredKey);
      if (!isHovered && !isNeighbor) {
        display.color = "#d8d8d8";
        display.label = "";
      }
    }
    return display;
  });

  renderer.setSetting("edgeReducer", (edge, attrs) => {
    const display = { ...attrs };
    if (selection.hoveredId !== null) {
      const hoveredKey = String(selection.hoveredId);
      const extremities = graph.extremities(edge);
      if (!extremities.includes(hoveredKey)) {
        display.hidden = true;
      }
    }
    return display;
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
          selection.selectedId = node.id;
          flyToNode(renderer, graph, String(node.id));
          resultsEl.innerHTML = "";
          searchInput.value = node.name;
          renderSidebar();
        });
        resultsEl.appendChild(item);
      }
    }, 100);
  });
}

bootstrap().catch((err) => {
  console.error(err);
});
