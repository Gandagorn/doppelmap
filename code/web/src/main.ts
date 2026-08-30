import Sigma from "sigma";
import { loadGraphData, buildGraphology } from "./graphData";
import { getDisplayMode } from "./sigmaSetup";
import { searchNames } from "./search";
import { flyToNode, getSidebarData, escapeHtml } from "./interactions";
import { DIM_NODE_COLOR } from "./theme";
import { fetchWikipediaPhoto } from "./wikipediaPhoto";
import type { GraphData } from "./types";

// Each popularity level is its own precomputed dataset (own kNN graph, own
// layout) rather than a filtered view of one big graph -- see
// build_dataset.py's POPULARITY_LEVELS. Order matches the slider's 4 steps.
const LEVEL_FILES = ["graph-all.json", "graph-top50.json", "graph-top20.json", "graph-top5.json"];
const LEVEL_LABELS = ["Show all", "Top 50%", "Top 20%", "Top 5%"];

async function bootstrap() {
  const container = document.getElementById("graph-container");
  if (!container) throw new Error("#graph-container not found in DOM");
  const isDark = window.matchMedia?.("(prefers-color-scheme: dark)").matches ?? false;

  const sidebarEl = document.getElementById("sidebar") as HTMLElement;
  const searchInput = document.getElementById("search") as HTMLInputElement;
  const resultsEl = document.getElementById("search-results") as HTMLDivElement;
  const popularitySlider = document.getElementById("popularity-slider") as HTMLInputElement;
  const popularityLabelEl = document.getElementById("popularity-label") as HTMLSpanElement;

  const dataCache = new Map<number, GraphData>();

  // Total UI state: which node is selected (sidebar open) and which is
  // hovered (dims everything else). Just a plain object mutated in place —
  // two fields don't need a reducer.
  const selection: { selectedId: number | null; hoveredId: number | null } = {
    selectedId: null,
    hoveredId: null,
  };

  // Reassigned by loadLevel() on every slider change; always assigned
  // before any handler that reads them can actually run (loadLevel
  // completes once before bootstrap() returns, and nothing before that
  // is interactive).
  let data!: GraphData;
  let graph!: ReturnType<typeof buildGraphology>;
  let renderer!: Sigma;

  function renderSidebar() {
    if (selection.selectedId === null) {
      sidebarEl.hidden = true;
      sidebarEl.innerHTML = "";
      return;
    }
    const info = getSidebarData(data, selection.selectedId);
    const localThumbSrc = `${import.meta.env.BASE_URL}data/${info.thumb}`;
    sidebarEl.hidden = false;
    sidebarEl.innerHTML = `
      <img id="sidebar-photo" src="${escapeHtml(localThumbSrc)}" width="96" height="96" alt="${escapeHtml(info.name)}" />
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

    // Instant paint with the local placeholder above; swap in the real
    // photo once (if) it resolves. Guard against the user having selected
    // a different node (or switched levels) before this fetch comes back.
    const requestedId = selection.selectedId;
    fetchWikipediaPhoto(info.name).then((url) => {
      if (url === null || selection.selectedId !== requestedId) return;
      const img = document.getElementById("sidebar-photo") as HTMLImageElement | null;
      const attrEl = sidebarEl.querySelector<HTMLParagraphElement>(".attr");
      if (img) img.src = url;
      if (attrEl) attrEl.textContent = "Photo: Wikipedia";
    });
  }

  function selectNode(id: number) {
    selection.selectedId = id;
    resultsEl.innerHTML = "";
    flyToNode(renderer, String(id));
    renderSidebar();
  }

  async function loadLevel(levelIndex: number) {
    // A selection/search from the previous level doesn't necessarily exist
    // in the new one -- close the sidebar and clear search rather than
    // show something wrong.
    selection.selectedId = null;
    selection.hoveredId = null;
    resultsEl.innerHTML = "";
    searchInput.value = "";
    renderSidebar();

    let levelData = dataCache.get(levelIndex);
    if (!levelData) {
      levelData = await loadGraphData(
        `${import.meta.env.BASE_URL}data/${LEVEL_FILES[levelIndex]}`
      );
      dataCache.set(levelIndex, levelData);
    }
    data = levelData;

    if (renderer) renderer.kill();
    graph = buildGraphology(data, isDark);
    renderer = new Sigma(graph, container as HTMLElement, {
      labelRenderedSizeThreshold: 0,
    });

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

    renderer.setSetting("nodeReducer", (nodeId, attrs) => {
      const display = { ...attrs };
      const mode = getDisplayMode(renderer.getCamera().ratio);
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

    popularityLabelEl.textContent = `${LEVEL_LABELS[levelIndex]} (${data.nodes.length})`;
  }

  window.addEventListener("keydown", (evt) => {
    if (evt.key === "Escape") {
      selection.selectedId = null;
      resultsEl.innerHTML = "";
      renderSidebar();
    }
  });

  popularitySlider.addEventListener("input", () => {
    loadLevel(Number(popularitySlider.value)).catch((err) => console.error(err));
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

  await loadLevel(Number(popularitySlider.value));
}

bootstrap().catch((err) => {
  console.error(err);
});
