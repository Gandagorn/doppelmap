import Sigma from "sigma";
import { loadGraphData, buildGraphology } from "./graphData";
import { getDisplayMode } from "./sigmaSetup";
import { searchNames } from "./search";
import { flyToNode, getSidebarData, formatSimilarity, escapeHtml } from "./interactions";
import { DIM_NODE_COLOR, SELECTED_NODE_COLOR } from "./theme";
import { fetchWikipediaInfo } from "./wikipediaPhoto";
import type { GraphData, GraphNode } from "./types";

// Each popularity level is its own precomputed dataset (own kNN graph, own
// layout) rather than a filtered view of one big graph -- see
// build_dataset.py's POPULARITY_LEVELS. Order matches the slider's 4 steps.
const LEVEL_FILES = ["graph-all.json", "graph-top50.json", "graph-top20.json", "graph-top5.json"];
const LEVEL_LABELS = ["Show all", "Top 50%", "Top 20%", "Top 5%"];
const DEFAULT_LEVEL_INDEX = 2;

const WALK_STEP_MS = 2000;

async function bootstrap() {
  const container = document.getElementById("graph-container");
  if (!container) throw new Error("#graph-container not found in DOM");
  const isDark = window.matchMedia?.("(prefers-color-scheme: dark)").matches ?? false;

  const sidebarEl = document.getElementById("sidebar") as HTMLElement;
  const toolbarEl = document.getElementById("toolbar") as HTMLElement;
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
  const pairView = document.getElementById("pair-view") as HTMLElement;
  const pairBody = document.getElementById("pair-body") as HTMLElement;
  const pairClose = document.getElementById("pair-close") as HTMLButtonElement;

  const initialParams = new URLSearchParams(location.search);
  const levelParam = initialParams.get("level");
  if (levelParam !== null) {
    // Number(null) is 0, not NaN -- checking this only when the param is
    // actually present, rather than parsing null itself, matters here: a
    // bare URL with no ?level at all was otherwise silently overriding the
    // HTML default with level 0.
    const initialLevel = Number(levelParam);
    if (Number.isInteger(initialLevel) && initialLevel >= 0 && initialLevel < LEVEL_FILES.length) {
      popularitySlider.value = String(initialLevel);
    }
  }
  const initialPersonName = initialParams.get("person");

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

  // Reflects current state (fame level + selection) into the URL via
  // replaceState -- no new history entry per click, but the address bar
  // always has a link a user can copy to share exactly what they're
  // looking at.
  function updateUrl() {
    const params = new URLSearchParams();
    const level = Number(popularitySlider.value);
    if (level !== DEFAULT_LEVEL_INDEX) params.set("level", String(level));
    if (selection.selectedId !== null) {
      const name = data.nodes.find((n) => n.id === selection.selectedId)?.name;
      if (name) params.set("person", name);
    }
    const query = params.toString();
    history.replaceState(null, "", query ? `?${query}` : location.pathname);
  }

  function updateWalkToggleLabel() {
    walkToggle.textContent = walk.active ? "⏸ Stop Walking" : "▶ Walk the Graph";
    walkToggle.classList.toggle("walking", walk.active);
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
      updateUrl();
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
                <button class="row-compare" type="button" data-compare="${s.id}"
                        data-w="${s.weight}" title="Compare faces side by side"
                        aria-label="Compare with ${escapeHtml(s.name)}">⇄</button>
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
      const compare = li.querySelector<HTMLButtonElement>("button[data-compare]");
      compare?.addEventListener("click", (evt) => {
        // Without this the row's own click handler also fires and navigates
        // away from the person we're trying to compare against.
        evt.stopPropagation();
        hoverPreview.hidden = true;
        if (selection.selectedId === null) return;
        showPair(selection.selectedId, Number(compare.dataset.compare), Number(compare.dataset.w));
      });
    });

    updateUrl();

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

  /** Centre of the canvas area not covered by UI, in viewport pixels.
   *
   *  Measured from the live elements rather than derived from the
   *  breakpoint, so it follows the CSS: the sidebar is a right-hand panel
   *  on desktop and a bottom sheet on mobile, and its height there (55vh)
   *  would otherwise hide anything the camera centred. */
  function visibleCentre(): { x: number; y: number } {
    const { width, height } = renderer.getDimensions();
    const top = toolbarEl.getBoundingClientRect().bottom;
    if (sidebarEl.hidden) return { x: width / 2, y: (top + height) / 2 };

    const sidebar = sidebarEl.getBoundingClientRect();
    // A bottom sheet spans the full width; the desktop sidebar is docked
    // right and leaves the canvas to its left.
    if (sidebar.width >= width * 0.9) {
      return { x: width / 2, y: (top + sidebar.top) / 2 };
    }
    return { x: sidebar.left / 2, y: (top + height) / 2 };
  }

  function selectNode(id: number) {
    selection.selectedId = id;
    resultsEl.innerHTML = "";
    // Sidebar first: visibleCentre() measures it, so it has to be on
    // screen and laid out before the camera target is computed.
    renderSidebar();
    flyToNode(renderer, String(id), visibleCentre());
    // nodeReducer/edgeReducer output is cached and only re-evaluated on
    // refresh() -- the mouse hovering the graph triggers that incidentally
    // (enterNode/leaveNode both call it), which is why a manual click
    // happened to look right, but any selection made without the mouse
    // over the destination node (search, sidebar/dashboard clicks, walk
    // steps) left the highlight frozen on whatever it was before.
    renderer.refresh();
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
    renderer.refresh();
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

  // The whole point of the site is "these two look alike", but until now
  // you could only ever see one face at a time -- a pair was rendered as
  // one photo plus the other person's name as text. This shows both at
  // once, which is also the thing worth screenshotting.
  function showPair(idA: number, idB: number, weight: number) {
    const nodesById = new Map(data.nodes.map((n) => [n.id, n]));
    const a = nodesById.get(idA);
    const b = nodesById.get(idB);
    if (!a || !b) return;

    const side = (n: GraphNode) => `
      <figure class="pair-side">
        <img src="${escapeHtml(`${import.meta.env.BASE_URL}data/${n.thumb}`)}"
             alt="${escapeHtml(n.name)}" data-name="${escapeHtml(n.name)}" />
        <figcaption>${escapeHtml(n.name)}</figcaption>
        <button class="pair-goto" type="button" data-goto="${n.id}">View on map</button>
      </figure>`;

    pairBody.innerHTML = `
      ${side(a)}
      <div class="pair-score">
        <strong>${formatSimilarity(weight)}</strong>
        <span>similarity</span>
      </div>
      ${side(b)}
    `;
    pairView.hidden = false;

    // Same instant-placeholder-then-swap pattern the sidebar uses.
    pairBody.querySelectorAll<HTMLImageElement>("img[data-name]").forEach((img) => {
      fetchWikipediaInfo(img.dataset.name ?? "").then((wiki) => {
        if (wiki.photoUrl) img.src = wiki.photoUrl;
      });
    });
    pairBody.querySelectorAll<HTMLButtonElement>("button[data-goto]").forEach((btn) => {
      btn.addEventListener("click", () => {
        closePair();
        selectNodeManually(Number(btn.dataset.goto));
      });
    });
  }

  function closePair() {
    pairView.hidden = true;
    pairBody.innerHTML = "";
  }

  function renderDashboard() {
    const nodesById = new Map(data.nodes.map((n) => [n.id, n]));
    const topPairs = [...data.edges].sort((a, b) => b[2] - a[2]).slice(0, 50);
    dashboardListEl.innerHTML = topPairs
      .map(([a, b, w]) => {
        const nameA = nodesById.get(a)?.name ?? "Unknown";
        const nameB = nodesById.get(b)?.name ?? "Unknown";
        return `<li data-a="${a}" data-b="${b}" data-w="${w}">${escapeHtml(nameA)} ↔ ${escapeHtml(nameB)} (${formatSimilarity(w)})</li>`;
      })
      .join("");
    dashboardListEl.querySelectorAll<HTMLLIElement>("li[data-a]").forEach((li) => {
      li.addEventListener("click", () => {
        // A dashboard row *is* a pair -- it used to just select one half and
        // drop the other, which threw away the comparison being pointed at.
        showPair(Number(li.dataset.a), Number(li.dataset.b), Number(li.dataset.w));
        dashboardEl.hidden = true;
      });
    });
  }

  async function loadLevel(levelIndex: number) {
    // Node ids are just a per-level array index (see build_dataset.py), not
    // stable across levels -- carry the selection over by name instead, and
    // only actually drop it if this level filtered that person out.
    const previouslySelectedName =
      selection.selectedId !== null && data
        ? (data.nodes.find((n) => n.id === selection.selectedId)?.name ?? null)
        : null;

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

      // The selected node itself gets its own look, independent of hover --
      // otherwise it's only as visible as "not dimmed", easy to lose track
      // of at a glance, especially at low degree (small size).
      if (selection.selectedId !== null && nodeId === String(selection.selectedId)) {
        display.color = SELECTED_NODE_COLOR;
        display.size = attrs.size + 2;
        display.zIndex = 1;
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

    if (previouslySelectedName) {
      const stillPresent = data.nodes.find((n) => n.name === previouslySelectedName);
      if (stillPresent) selectNode(stillPresent.id);
    }
  }

  window.addEventListener("keydown", (evt) => {
    if (evt.key !== "Escape") return;
    // Escape closes the topmost thing first rather than always clearing the
    // selection underneath the pair view.
    if (!pairView.hidden) {
      closePair();
    } else {
      deselectManually();
    }
  });

  pairClose.addEventListener("click", closePair);
  pairView.addEventListener("click", (evt) => {
    // Backdrop only -- clicks inside the card shouldn't dismiss it.
    if (evt.target === pairView) closePair();
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
  // Deferred one frame: right after `new Sigma(...)`, in the same tick,
  // its display data (node positions in camera space) hasn't been computed
  // by an actual render pass yet, so flyToNode's camera.animate() targets
  // stale/default coordinates instead of the node's real position -- the
  // camera visibly moves but doesn't end up centered on anyone. A later
  // click/search selection never hits this because the page has already
  // rendered at least one frame by the time a user can interact with it.
  requestAnimationFrame(() => {
    if (initialPersonName) {
      const match = data.nodes.find((n) => n.name === initialPersonName);
      if (match) {
        selectNode(match.id);
        return;
      }
    }
    // Land on someone rather than an empty map, but only once, at startup
    // -- no ongoing auto-walk.
    selectNode(pickRandomNodeId());
  });
}

bootstrap().catch((err) => {
  console.error(err);
});
