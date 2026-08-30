import Sigma from "sigma";
import { loadGraphData, buildGraphology } from "./graphData";
import { getDisplayMode } from "./sigmaSetup";
import { searchNames } from "./search";
import { flyToNode, getSidebarData, escapeHtml } from "./interactions";
import { DIM_NODE_COLOR } from "./theme";

async function bootstrap() {
  const data = await loadGraphData(`${import.meta.env.BASE_URL}data/graph.json`);
  const isDark = window.matchMedia?.("(prefers-color-scheme: dark)").matches ?? false;
  const graph = buildGraphology(data, isDark);

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
  const searchInput = document.getElementById("search") as HTMLInputElement;
  const resultsEl = document.getElementById("search-results") as HTMLDivElement;

  function selectNode(id: number) {
    selection.selectedId = id;
    resultsEl.innerHTML = "";
    flyToNode(renderer, String(id));
    renderSidebar();
  }

  function renderSidebar() {
    if (selection.selectedId === null) {
      sidebarEl.hidden = true;
      sidebarEl.innerHTML = "";
      return;
    }
    const info = getSidebarData(data, selection.selectedId);
    const photoSrc = info.photo ?? `${import.meta.env.BASE_URL}data/${info.thumb}`;
    sidebarEl.hidden = false;
    sidebarEl.innerHTML = `
      <img src="${escapeHtml(photoSrc)}" width="96" height="96" alt="${escapeHtml(info.name)}" />
      <h2>${escapeHtml(info.name)}</h2>
      <p class="attr">${escapeHtml(info.attr)}</p>
      <ul class="similar-list">
        ${info.similar
          .map((s) => `<li data-id="${s.id}">${escapeHtml(s.name)} — ${s.percent}</li>`)
          .join("")}
      </ul>
    `;
    sidebarEl.querySelectorAll<HTMLLIElement>("li[data-id]").forEach((li) => {
      li.addEventListener("click", () => {
        selectNode(Number(li.dataset.id));
      });
    });
  }

  renderer.on("clickNode", ({ node }) => {
    selectNode(Number(node));
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
    resultsEl.innerHTML = "";
    renderSidebar();
  });

  window.addEventListener("keydown", (evt) => {
    if (evt.key === "Escape") {
      selection.selectedId = null;
      resultsEl.innerHTML = "";
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
        display.color = DIM_NODE_COLOR;
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

  // nodeReducer's output is cached and only re-runs on refresh() (which
  // camera pan/zoom does NOT trigger on its own), so without this the
  // zoom-dependent label mode would only update by coincidence, e.g. on
  // hover. Only refresh when the mode actually flips, to avoid a full
  // reprocess on every pan/zoom tick.
  let lastDisplayMode = getDisplayMode(renderer.getCamera().ratio);
  renderer.getCamera().on("updated", () => {
    const mode = getDisplayMode(renderer.getCamera().ratio);
    if (mode !== lastDisplayMode) {
      lastDisplayMode = mode;
      renderer.refresh({ skipIndexation: true });
    }
  });

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
          searchInput.value = node.name;
          selectNode(node.id);
        });
        resultsEl.appendChild(item);
      }
    }, 100);
  });
}

bootstrap().catch((err) => {
  console.error(err);
});
