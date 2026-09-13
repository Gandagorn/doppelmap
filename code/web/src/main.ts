import Sigma from "sigma";
import { loadGraphData, buildGraphology } from "./graphData";
import { getDisplayMode } from "./sigmaSetup";
import { searchNames } from "./search";
import type { SidebarRow } from "./interactions";
import {
  flyToNode, getSidebarData, formatSimilarity, escapeHtml, groupByResemblance,
  resemblanceBand,
} from "./interactions";
import { DIM_NODE_COLOR, FADED_EDGE_COLOR, SELECTED_NODE_COLOR } from "./theme";
import { fetchWikipediaInfo } from "./wikipediaPhoto";
import { faceCropStyle, loadPhotos, photoUrl, photosFor } from "./photos";
import type { GraphData, GraphNode, PhotoRef } from "./types";

// One graph, not a stack of popularity levels. The fame slider filtered on
// how many photos a person had, which was only ever a proxy for fame, and
// the Commons collector gives everyone a comparable handful.
const GRAPH_FILE = "graph.json";
const PHOTOS_FILE = "photos.json";

const WALK_STEP_MS = 2000;

async function bootstrap() {
  const container = document.getElementById("graph-container");
  if (!container) throw new Error("#graph-container not found in DOM");
  const isDark = window.matchMedia?.("(prefers-color-scheme: dark)").matches ?? false;

  const sidebarEl = document.getElementById("sidebar") as HTMLElement;
  const toolbarEl = document.getElementById("toolbar") as HTMLElement;
  const searchInput = document.getElementById("search") as HTMLInputElement;
  const resultsEl = document.getElementById("search-results") as HTMLDivElement;
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

  const initialPersonName = new URLSearchParams(location.search).get("person");

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

  /** An <img> cropped to a person's face, or a lettered placeholder.

   *  photos.json arrives after the graph, so the first sidebar opened on a
   *  cold load has no crop to show yet -- hence the initials fallback
   *  rather than an empty box. */
  function faceHtml(id: number, name: string, cls: string, width: number): string {
    const photo = photosFor(id)?.[0];
    const size = `width:${width}px;height:${width}px`;
    if (!photo) {
      const initials = name.split(" ").map((w) => w[0]).join("").slice(0, 2);
      return `<span class="${cls} face-fallback" style="${size}"
                    aria-label="${escapeHtml(name)}">${escapeHtml(initials)}</span>`;
    }
    return `<span class="${cls}" style="${faceCropStyle(photo, 0.55, width)}${size}"
                  role="img" aria-label="${escapeHtml(name)}"></span>`;
  }

  function renderSidebar() {
    if (selection.selectedId === null) {
      sidebarEl.hidden = true;
      sidebarEl.innerHTML = "";
      updateUrl();
      return;
    }
  /** The similar list, grouped under plain-language headings.
   *
   *  Scores below the floor are not shown at all: they are indistinguishable
   *  from each other and from nothing. Someone with no match above it still
   *  gets told who their closest is, because an empty panel reads as a
   *  broken page rather than as an answer.
   */
  const renderSimilar = (info: ReturnType<typeof getSidebarData>) => {
    const row = (s: SidebarRow) => `
      <li data-id="${s.id}">
        ${faceHtml(s.id, s.name, "row-thumb", 36)}
        <span class="row-name">${escapeHtml(s.name)}</span>
        <span class="row-percent">${s.percent}</span>
        <button class="row-compare" type="button" data-compare="${s.id}"
                data-w="${s.weight}" data-mine="${s.myPhoto}" data-theirs="${s.theirPhoto}"
                title="Compare the two closest photos"
                aria-label="Compare with ${escapeHtml(s.name)}">⇄</button>
      </li>`;

    const groups = groupByResemblance(info.similar);
    if (!groups.length) {
      const closest = info.similar[0];
      return `<p class="no-resemblance">No one in this dataset looks much like
        ${escapeHtml(info.name)}.${closest
          ? ` The closest is ${escapeHtml(closest.name)}, at ${closest.percent}.`
          : ""}</p>`;
    }
    return groups
      .map(
        (g) => `
        <h3 class="resemblance-band" data-band="${g.band.slug}">${escapeHtml(g.band.label)}</h3>
        <ul class="similar-list">${g.entries.map(row).join("")}</ul>`
      )
      .join("");
  };

    const info = getSidebarData(data, selection.selectedId);
    const photo = photosFor(info.id)?.[0];
    sidebarEl.hidden = false;
    sidebarEl.innerHTML = `
      <div id="sidebar-photo">${faceHtml(info.id, info.name, "face", 160)}</div>
      <h2>${escapeHtml(info.name)}</h2>
      <p class="attr"></p>
      <p class="known-for" hidden></p>
      ${photo ? `<p class="credit">${escapeHtml(photo.c || "Wikimedia Commons")}${
          photo.l ? ` · ${escapeHtml(photo.l)}` : ""}</p>` : ""}
      ${renderSimilar(info)}
    `;

    sidebarEl.querySelectorAll<HTMLLIElement>("li[data-id]").forEach((li) => {
      li.addEventListener("click", () => selectNodeManually(Number(li.dataset.id)));

      // Enlarged preview of whatever this row already shows -- no extra
      // request, since it is the same Commons image at a bigger width.
      li.addEventListener("mouseenter", () => {
        const p = photosFor(Number(li.dataset.id))?.[0];
        if (!p) return;
        hoverPreview.setAttribute("style", faceCropStyle(p, 0.55, 200));
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
        // away from the person we are trying to compare against.
        evt.stopPropagation();
        hoverPreview.hidden = true;
        if (selection.selectedId === null) return;
        showPair(
          selection.selectedId, Number(compare.dataset.compare), Number(compare.dataset.w),
          Number(compare.dataset.mine), Number(compare.dataset.theirs),
        );
      });
    });

    updateUrl();

    // Wikipedia supplies the one-line description and a link out. The photo
    // itself now comes from our own data, so this no longer has to race to
    // replace a placeholder. Guard against the selection moving on before
    // the lookup returns.
    const requestedId = selection.selectedId;
    fetchWikipediaInfo(info.name).then((wiki) => {
      if (selection.selectedId !== requestedId) return;
      const attrEl = sidebarEl.querySelector<HTMLParagraphElement>(".attr");
      if (attrEl && wiki.pageUrl) {
        attrEl.innerHTML = `<a href="${escapeHtml(wiki.pageUrl)}" target="_blank" rel="noopener noreferrer">Wikipedia</a>`;
      }
      if (wiki.description) {
        const descEl = sidebarEl.querySelector<HTMLParagraphElement>(".known-for");
        if (descEl) {
          descEl.textContent = wiki.description;
          descEl.hidden = false;
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
  function showPair(
    idA: number, idB: number, weight: number, photoA = 0, photoB = 0
  ) {
    const nodesById = new Map(data.nodes.map((n) => [n.id, n]));
    const a = nodesById.get(idA);
    const b = nodesById.get(idB);
    if (!a || !b) return;

    // The specific photographs the model matched, chosen at build time --
    // not two arbitrary portraits. Seeing the actual pair is what makes a
    // resemblance judgeable rather than a number to take on trust.
    const side = (n: GraphNode, which: number) => {
      const photo: PhotoRef | undefined = photosFor(n.id)?.[which] ?? photosFor(n.id)?.[0];
      const media = photo
        ? `<span class="pair-face" role="img" aria-label="${escapeHtml(n.name)}"
                 style="${faceCropStyle(photo, 0.75, 190)}"></span>`
        : `<span class="face-fallback pair-fallback">${escapeHtml(
            n.name.split(" ").map((w) => w[0]).join("").slice(0, 2))}</span>`;
      const credit = photo && (photo.c || photo.l)
        ? `<span class="pair-credit">${escapeHtml(
            [photo.c, photo.l].filter(Boolean).join(" · "))}</span>`
        : "";
      return `
        <figure class="pair-side">
          ${media}
          <figcaption>${escapeHtml(n.name)}</figcaption>
          ${credit}
          <button class="pair-goto" type="button" data-goto="${n.id}">View on map</button>
        </figure>`;
    };

    // The same wording the sidebar groups by, so a score means one thing
    // everywhere it appears.
    const band = resemblanceBand(weight);

    pairBody.innerHTML = `
      ${side(a, photoA)}
      <div class="pair-score">
        <strong>${formatSimilarity(weight)}</strong>
        ${band
          ? `<span class="pair-band" data-band="${band.slug}">${escapeHtml(band.label)}</span>`
          : `<span>similarity</span>`}
      </div>
      ${side(b, photoB)}
    `;
    pairView.hidden = false;

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
        const match = (data.similar[String(a)] ?? []).find((e) => e[0] === b);
        return `<li data-a="${a}" data-b="${b}" data-w="${w}" data-mine="${match?.[2] ?? 0}" data-theirs="${match?.[3] ?? 0}">${escapeHtml(nameA)} ↔ ${escapeHtml(nameB)} (${formatSimilarity(w)})</li>`;
      })
      .join("");
    dashboardListEl.querySelectorAll<HTMLLIElement>("li[data-a]").forEach((li) => {
      li.addEventListener("click", () => {
        // A dashboard row *is* a pair -- it used to just select one half and
        // drop the other, which threw away the comparison being pointed at.
        showPair(Number(li.dataset.a), Number(li.dataset.b), Number(li.dataset.w),
                 Number(li.dataset.mine), Number(li.dataset.theirs));
        dashboardEl.hidden = true;
      });
    });
  }

  async function loadGraph() {
    data = await loadGraphData(`${import.meta.env.BASE_URL}data/${GRAPH_FILE}`);
    // Kicked off, not awaited: photos.json is several times the size of the
    // graph and only needed once someone opens a person, so the map appears
    // without waiting for it.
    loadPhotos(`${import.meta.env.BASE_URL}data/${PHOTOS_FILE}`).then(() => {
      if (selection.selectedId !== null) renderSidebar();
    });

    graph = buildGraphology(data, isDark);
    renderer = new Sigma(graph, container as HTMLElement, {
      // Our nodes are deliberately small (1.5-4px), so Sigma's default
      // size threshold would suppress every label. Zero lets them all
      // qualify; the grid settings below then decide how many actually
      // fit, rather than painting overlapping names on top of each other.
      labelRenderedSizeThreshold: 0,
      labelGridCellSize: 250,
      labelDensity: 0.6,
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
          // Dimmed, not hidden. Hiding every non-incident edge blanked the
          // entire map whenever anything was selected -- and since startup
          // auto-selects someone, that was the first thing anyone saw.
          display.color = FADED_EDGE_COLOR;
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

    renderDashboard();
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

  await loadGraph();
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
