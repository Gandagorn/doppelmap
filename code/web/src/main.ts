import Sigma from "sigma";
import { loadGraphData, buildGraphology } from "./graphData";
import { getDisplayMode } from "./sigmaSetup";
import { searchNames } from "./search";
import { flyToNode, getSidebarData, formatSimilarity, escapeHtml } from "./interactions";
import { DIM_NODE_COLOR } from "./theme";
import { fetchWikipediaInfo } from "./wikipediaPhoto";
import type { GraphData } from "./types";

// Each popularity level is its own precomputed dataset (own kNN graph, own
// layout) rather than a filtered view of one big graph -- see
// build_dataset.py's POPULARITY_LEVELS. Order matches the slider's 4 steps.
const LEVEL_FILES = ["graph-all.json", "graph-top50.json", "graph-top20.json", "graph-top5.json"];
const LEVEL_LABELS = ["Show all", "Top 50%", "Top 20%", "Top 5%"];

const WALK_STEP_MS = 2000;

async function bootstrap() {
  const container = document.getElementById("graph-container");
  if (!container) throw new Error("#graph-container not found in DOM");
  const isDark = window.matchMedia?.("(prefers-color-scheme: dark)").matches ?? false;

  const sidebarEl = document.getElementById("sidebar") as HTMLElement;
  const searchInput = document.getElementById("search") as HTMLInputElement;
  const resultsEl = document.getElementById("search-results") as HTMLDivElement;
  const popularitySlider = document.getElementById("popularity-slider") as HTMLInputElement;
  const popularityLabelEl = document.getElementById("popularity-label") as HTMLSpanElement;
  const dashboardToggle = document.getElementById("dashboard-toggle") as HTMLButtonElement;
  const dashboardEl = document.getElementById("dashboard") as HTMLElement;
  const dashboardClose = document.getElementById("dashboard-close") as HTMLButtonElement;
  const dashboardListEl = document.getElementById("dashboard-list") as HTMLOListElement;
  const hoverPreview = document.getElementById("hover-preview") as HTMLImageElement;
  const aboutToggle = document.getElementById("about-toggle") as HTMLButtonElement;
  const aboutPanel = document.getElementById("about-panel") as HTMLElement;
  const aboutClose = document.getElementById("about-close") as HTMLButtonElement;
  const walkToggle = document.getElementById("walk-toggle") as HTMLButtonElement;

  const dataCache = new Map<number, GraphData>();

  // Total UI state: which node is selected (sidebar open) and which is
  // hovered (dims everything else). Just a plain object mutated in place —
  // two fields don't need a reducer.
  const selection: { selectedId: number | null; hoveredId: number | null } = {
    selectedId: null,
    hoveredId: null,
  };

  // "Walk the Graph": auto-advance from the selected node to its highest-
  // similarity not-yet-visited neighbor, repeating on a timer. visited
  // prevents cycles; the walk stops on its own at a dead end (every
  // neighbor already visited).
  const walk: { active: boolean; visited: Set<number>; timer: ReturnType<typeof setInterval> | undefined } = {
    active: false,
    visited: new Set(),
    timer: undefined,
  };

  // Reassigned by loadLevel() on every slider change; always assigned
  // before any handler that reads them can actually run (loadLevel
  // completes once before bootstrap() returns, and nothing before that
  // is interactive).
  let data!: GraphData;
  let graph!: ReturnType<typeof buildGraphology>;
  let renderer!: Sigma;

  function updateWalkToggleLabel() {
    walkToggle.textContent = walk.active ? "⏸ Stop Walking" : "▶ Walk the Graph";
  }

  function stopWalk() {
    walk.active = false;
    if (walk.timer !== undefined) {
      clearInterval(walk.timer);
      walk.timer = undefined;
    }
    updateWalkToggleLabel();
  }

  function renderSidebar() {
    if (selection.selectedId === null) {
      sidebarEl.hidden = true;
      sidebarEl.innerHTML = "";
      return;
    }
    const info = getSidebarData(data, selection.selectedId);
    const localThumbSrc = (thumb: string) => `${import.meta.env.BASE_URL}data/${thumb}`;
    sidebarEl.hidden = false;
    sidebarEl.innerHTML = `
      <img id="sidebar-photo" src="${escapeHtml(localThumbSrc(info.thumb))}" width="160" height="160" alt="${escapeHtml(info.name)}" />
      <h2>${escapeHtml(info.name)}</h2>
      <p class="attr">${escapeHtml(info.attr)}</p>
      <ul class="similar-list">
        ${info.similar
          .map(
            (s) => `
              <li data-id="${s.id}">
                <img class="row-thumb" src="${escapeHtml(localThumbSrc(s.thumb))}" width="36" height="36" alt="" />
                <span class="row-name">${escapeHtml(s.name)}</span>
                <span class="row-percent">${s.percent}</span>
              </li>`
          )
          .join("")}
      </ul>
    `;
    // Preload every neighbor's real photo as soon as the sidebar opens, so
    // it's already in wikipediaPhoto's cache (usually resolved outright)
    // by the time the user looks at or clicks one of these rows.
    sidebarEl.querySelectorAll<HTMLLIElement>("li[data-id]").forEach((li) => {
      li.addEventListener("click", () => {
        selectNodeManually(Number(li.dataset.id));
      });
      const name = li.querySelector<HTMLSpanElement>(".row-name")?.textContent ?? "";
      const img = li.querySelector<HTMLImageElement>(".row-thumb");
      if (name && img) {
        fetchWikipediaInfo(name).then((wiki) => {
          if (wiki.photoUrl) img.src = wiki.photoUrl;
        });
      }
      // Show a larger version of whatever the row currently displays
      // (placeholder or, usually by now, the real preloaded photo).
      li.addEventListener("mouseenter", () => {
        if (!img) return;
        hoverPreview.src = img.src;
        const rect = li.getBoundingClientRect();
        hoverPreview.style.top = `${Math.max(8, rect.top - 90)}px`;
        hoverPreview.style.left = `${rect.left - 216}px`;
        hoverPreview.hidden = false;
      });
      li.addEventListener("mouseleave", () => {
        hoverPreview.hidden = true;
      });
    });

    // Instant paint with the local placeholder above; swap in the real
    // photo/link once (if) it resolves. Guard against the user having
    // selected a different node (or switched levels) before this fetch
    // comes back.
    const requestedId = selection.selectedId;
    fetchWikipediaInfo(info.name).then((wiki) => {
      if (selection.selectedId !== requestedId) return;
      if (wiki.photoUrl) {
        const img = document.getElementById("sidebar-photo") as HTMLImageElement | null;
        if (img) img.src = wiki.photoUrl;
      }
      if (wiki.pageUrl) {
        const attrEl = sidebarEl.querySelector<HTMLParagraphElement>(".attr");
        if (attrEl) {
          attrEl.innerHTML = `Photo: <a href="${escapeHtml(wiki.pageUrl)}" target="_blank" rel="noopener noreferrer">Wikipedia</a>`;
        }
      }
    });
  }

  function selectNode(id: number) {
    selection.selectedId = id;
    resultsEl.innerHTML = "";
    flyToNode(renderer, String(id));
    renderSidebar();
  }

  // Manual interactions (clicking a node, a search result, a similar-list
  // entry, a dashboard entry) stop any running walk -- the user taking
  // control should end the automated tour rather than fight it.
  function selectNodeManually(id: number) {
    stopWalk();
    selectNode(id);
  }

  function deselectManually() {
    stopWalk();
    selection.selectedId = null;
    resultsEl.innerHTML = "";
    renderSidebar();
  }

  function stepWalk() {
    if (selection.selectedId === null) {
      stopWalk();
      return;
    }
    const info = getSidebarData(data, selection.selectedId);
    const next = info.similar.find((s) => !walk.visited.has(s.id));
    if (!next) {
      stopWalk();
      return;
    }
    walk.visited.add(next.id);
    selectNode(next.id);
  }

  function startWalk() {
    if (selection.selectedId === null) return;
    walk.active = true;
    walk.visited = new Set([selection.selectedId]);
    updateWalkToggleLabel();
    stepWalk();
    walk.timer = setInterval(stepWalk, WALK_STEP_MS);
  }

  function pickRandomNodeId(): number {
    return data.nodes[Math.floor(Math.random() * data.nodes.length)].id;
  }

  function renderDashboard() {
    const nodesById = new Map(data.nodes.map((n) => [n.id, n]));
    const topPairs = [...data.edges].sort((a, b) => b[2] - a[2]).slice(0, 50);
    dashboardListEl.innerHTML = topPairs
      .map(([a, b, w]) => {
        const nameA = nodesById.get(a)?.name ?? "Unknown";
        const nameB = nodesById.get(b)?.name ?? "Unknown";
        return `<li data-id="${a}">${escapeHtml(nameA)} ↔ ${escapeHtml(nameB)} (${formatSimilarity(w)})</li>`;
      })
      .join("");
    dashboardListEl.querySelectorAll<HTMLLIElement>("li[data-id]").forEach((li) => {
      li.addEventListener("click", () => {
        selectNodeManually(Number(li.dataset.id));
        dashboardEl.hidden = true;
      });
    });
  }

  async function loadLevel(levelIndex: number) {
    // A selection/search from the previous level doesn't necessarily exist
    // in the new one -- close the sidebar and clear search rather than
    // show something wrong.
    stopWalk();
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
      selectNodeManually(Number(node));
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
      deselectManually();
    });

    renderer.setSetting("nodeReducer", (nodeId, attrs) => {
      const display = { ...attrs };
      const mode = getDisplayMode(renderer.getCamera().ratio);
      if (mode === "dot") display.label = "";

      // A click "sticks" the highlight even after the mouse moves away;
      // hovering a (different) node temporarily previews its neighbors on
      // top of that.
      const highlightId = selection.hoveredId ?? selection.selectedId;
      if (highlightId !== null) {
        const highlightKey = String(highlightId);
        const isHighlighted = nodeId === highlightKey;
        const isNeighbor = graph.areNeighbors(nodeId, highlightKey);
        if (!isHighlighted && !isNeighbor) {
          display.color = DIM_NODE_COLOR;
          display.label = "";
        }
      }
      return display;
    });

    renderer.setSetting("edgeReducer", (edge, attrs) => {
      const display = { ...attrs };
      const highlightId = selection.hoveredId ?? selection.selectedId;
      if (highlightId !== null) {
        const highlightKey = String(highlightId);
        const extremities = graph.extremities(edge);
        if (!extremities.includes(highlightKey)) {
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

    popularityLabelEl.textContent = LEVEL_LABELS[levelIndex];
    renderDashboard();
  }

  window.addEventListener("keydown", (evt) => {
    if (evt.key === "Escape") {
      deselectManually();
    }
  });

  popularitySlider.addEventListener("input", () => {
    loadLevel(Number(popularitySlider.value)).catch((err) => console.error(err));
  });

  walkToggle.addEventListener("click", () => {
    if (walk.active) {
      stopWalk();
      return;
    }
    if (selection.selectedId === null) {
      selectNode(pickRandomNodeId());
    }
    startWalk();
  });

  aboutToggle.addEventListener("click", () => {
    aboutPanel.hidden = !aboutPanel.hidden;
  });
  aboutClose.addEventListener("click", () => {
    aboutPanel.hidden = true;
  });

  dashboardToggle.addEventListener("click", () => {
    dashboardEl.hidden = !dashboardEl.hidden;
  });
  dashboardClose.addEventListener("click", () => {
    dashboardEl.hidden = true;
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
          selectNodeManually(node.id);
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
