import Sigma from "sigma";

import { createNodeImageProgram } from "@sigma/node-image";
import { setFaceImage, loadGraphData, buildGraphology } from "./graphData";
import { faceCount, requestFaces, releaseExcept } from "./faceTiles";
import { edgeColorForStrength } from "./theme";
import { getDisplayMode } from "./sigmaSetup";
import { searchNames } from "./search";
import type { SidebarRow } from "./interactions";
import {
  flyToNode, getSidebarData, formatSimilarity, escapeHtml, groupByResemblance,
  rarityNote, resemblanceBand,
} from "./interactions";
import {
  DIM_NODE_COLOR, fadedEdgeColor, labelColor, SELECTED_NODE_COLOR,
} from "./theme";
import { fetchWikipediaInfo } from "./wikipediaPhoto";
import {
  faceCropStyle, loadPhotos, photoUrl, photosFor, rememberUploadedPhotos,
} from "./photos";
import type { GraphData, GraphNode, PhotoRef } from "./types";
import { renderShareCard } from "./shareCard";
import { creditHtml, faceHtml } from "./render";
import { castVote, describeTally, myVote, pairTally, rememberVote } from "./votes";

// One graph, not a stack of popularity levels.
const GRAPH_FILE = "graph.json";
const PHOTOS_FILE = "photos.json";

// Inline SVG rather than an emoji: an emoji renders differently on every platform and
// brings its own colour, which a quiet button should not.
const SHARE_GLYPH = `<svg viewBox="0 0 24 24" width="13" height="13"
    fill="none" stroke="currentColor" stroke-width="2"
    stroke-linecap="round" stroke-linejoin="round">
    <circle cx="18" cy="5" r="3"/><circle cx="6" cy="12" r="3"/>
    <circle cx="18" cy="19" r="3"/>
    <path d="M8.6 13.5l6.8 4M15.4 6.5l-6.8 4"/></svg>`;

/** A name on a dark rounded backing, so it reads over a face, an edge or the
 *  background. Sigma's own hover label paints dark text on white, which on
 *  this palette is unreadable, so the hovered name uses the same pill. */
function paintLabel(
  ctx: CanvasRenderingContext2D,
  data: { x: number; y: number; size: number; label: string | null },
  settings: { labelSize: number; labelFont: string; labelWeight: string },
  highlighted: boolean,
): void {
  if (!data.label) return;
  const size = highlighted ? settings.labelSize + 1 : settings.labelSize;
  ctx.font = `${highlighted ? 600 : settings.labelWeight} ${size}px ${settings.labelFont}`;
  const width = ctx.measureText(data.label).width;
  const x = data.x + data.size + 5;
  const y = data.y + size / 3;
  const padX = highlighted ? 7 : 5;
  const padY = highlighted ? 4 : 3;
  const boxX = x - padX;
  const boxY = y - size - padY + 2;
  const boxW = width + padX * 2;
  const boxH = size + padY * 2;
  const r = 6;

  ctx.beginPath();
  ctx.moveTo(boxX + r, boxY);
  ctx.arcTo(boxX + boxW, boxY, boxX + boxW, boxY + boxH, r);
  ctx.arcTo(boxX + boxW, boxY + boxH, boxX, boxY + boxH, r);
  ctx.arcTo(boxX, boxY + boxH, boxX, boxY, r);
  ctx.arcTo(boxX, boxY, boxX + boxW, boxY, r);
  ctx.closePath();
  ctx.fillStyle = highlighted ? "rgba(16, 22, 40, 0.94)" : "rgba(24, 31, 52, 0.8)";
  ctx.fill();
  if (highlighted) {
    ctx.strokeStyle = SELECTED_NODE_COLOR;
    ctx.lineWidth = 1.5;
    ctx.stroke();
  }

  ctx.fillStyle = highlighted ? "#fff6ea" : labelColor();
  ctx.fillText(data.label, x, y);
}

function drawNodeLabel(
  ctx: CanvasRenderingContext2D,
  data: { x: number; y: number; size: number; label: string | null },
  settings: { labelSize: number; labelFont: string; labelWeight: string },
): void {
  paintLabel(ctx, data, settings, false);
}

function drawNodeHover(
  ctx: CanvasRenderingContext2D,
  data: { x: number; y: number; size: number; label: string | null },
  settings: { labelSize: number; labelFont: string; labelWeight: string },
): void {
  paintLabel(ctx, data, settings, true);
}

async function bootstrap() {
  const container = document.getElementById("graph-container");
  if (!container) throw new Error("#graph-container not found in DOM");

  const sidebarEl = document.getElementById("sidebar") as HTMLElement;
  const toolbarEl = document.getElementById("toolbar") as HTMLElement;
  const searchInput = document.getElementById("search") as HTMLInputElement;
  const resultsEl = document.getElementById("search-results") as HTMLDivElement;
  const dashboardToggle = document.getElementById("dashboard-toggle") as HTMLButtonElement;
  const dashboardEl = document.getElementById("dashboard") as HTMLElement;
  const dashboardClose = document.getElementById("dashboard-close") as HTMLButtonElement;
  const dashboardListEl = document.getElementById("dashboard-list") as HTMLOListElement;
  const hoverPreview = document.getElementById("hover-preview") as HTMLImageElement;
  const hoverPair = document.getElementById("hover-pair") as HTMLElement;
  const aboutToggle = document.getElementById("about-toggle") as HTMLButtonElement;
  const aboutView = document.getElementById("about-view") as HTMLElement;
  const aboutClose = document.getElementById("about-close") as HTMLButtonElement;
  const pairView = document.getElementById("pair-view") as HTMLElement;
  const pairBody = document.getElementById("pair-body") as HTMLElement;
  const pairClose = document.getElementById("pair-close") as HTMLButtonElement;
  const pairNav = document.getElementById("pair-nav") as HTMLElement;
  const pairExplore = document.getElementById("pair-explore") as HTMLButtonElement;
  const pairPrev = document.getElementById("pair-prev") as HTMLButtonElement;
  const pairNext = document.getElementById("pair-next") as HTMLButtonElement;
  const pairPosition = document.getElementById("pair-position") as HTMLElement;

  // How far the camera may pull back.
  const MAX_ZOOM_OUT_RATIO = 1.5;
  // Face tiles are 128px, which is the point past which magnifying stops revealing
  // anything.
  const MAX_ZOOM_IN_RATIO = 0.03;
  // How far the camera centre may travel from the middle of the map.
  const PAN_REACH = 0.35;
  const PAN_MARGIN = 0.25;

  const initialParams = new URLSearchParams(location.search);
  const initialPersonName = initialParams.get("person");
  // A shared comparison carries the other person, so the link reopens the side-by-side
  // rather than dropping the recipient on a sidebar and asking them to find the pair
  // again.
  const initialVersusName = initialParams.get("vs");

  // Total UI state: which node is selected (sidebar open) and which is hovered (dims
  // everything else).
  let openPair: { a: number; b: number } | null = null;
  // Family is a different claim from resemblance, and face recognition conflates them:
  // it scores kinship-correlated bone structure, so 29% of matches at or above 30% are
  // relatives, and the top of the unfiltered ranking is the Sprouse twins, the Williams
  // sisters and two Jacksons.
  let hideRelatives = true;
  try {
    hideRelatives = localStorage.getItem("doppelmap.hideRelatives") !== "0";
  } catch {
    // Private windows and blocked site data both throw; the default holds.
  }

  // Which photograph each side is showing, so the shared card renders the pair that is
  // actually on screen rather than each person's first picture.
  let openPairPhotos = { a: 0, b: 0 };

  /** One comparison, enough to reopen it without recomputing anything. */
  type PairRef = { a: number; b: number; weight: number; photoA: number; photoB: number };

  // The ranking the open comparison was reached through, so Prev/Next step along the
  // list the person was actually reading -- the Top Pairs table, or one person's own
  // matches -- rather than some global order they never saw.
  let pairSeries: { items: PairRef[]; index: number } | null = null;
  // Resolves when photos.json has arrived.
  let photosReady: Promise<unknown> = Promise.resolve();
  // Whether the map draws faces.
  let showFaces = false;
  // Faces appear once the map is zoomed in past this camera ratio; further out
  // it is dots. Per-zoom rather than per-node, so the map is never half faces.
  const FACE_RATIO = 1.15;
  // A ceiling on faces held at once, nearest the middle of the screen first.
  // The whole collection fits under it; it exists so a larger dataset cannot
  // grow the atlas without limit.
  const MAX_TILES = 1800;
  // Released only past this, so an ordinary pan does not churn tiles in and out.
  const RELEASE_AT = 2000;
  let facesApplied = false;

  /** Hand the map its face tiles. */
  function enableFaces() {
    if (facesApplied || !showFaces) return;
    facesApplied = true;
    cutVisibleFaces();
  }

  /** Cut faces for the nodes currently on screen, and only those. */
  function cutVisibleFaces() {
    if (!facesApplied || !renderer) return;
    const { width, height } = renderer.getDimensions();
    const centre = { x: width / 2, y: height / 2 };
    // Every face, nearest the middle of the screen first: the map fills where
    // the eye is and then completes, so panning and zooming never wait.
    const order: { id: string; distance: number }[] = [];
    graph.forEachNode((id, attrs) => {
      const point = renderer.graphToViewport({ x: attrs.x as number, y: attrs.y as number });
      order.push({ id, distance: Math.hypot(point.x - centre.x, point.y - centre.y) });
    });
    order.sort((a, b) => a.distance - b.distance);
    // Cutting every face at once costs ~110 MB of texture, enough to stall a
    // tab that is short of memory. Only the nearest are held, and they are
    // released in batches rather than on every pan. A released tile stays in
    // the cache, so coming back needs no network.
    if (faceCount() > RELEASE_AT) {
      const keep = new Set(order.slice(0, MAX_TILES).map((o) => o.id));
      for (const id of releaseExcept(keep)) {
        if (graph.hasNode(id)) graph.setNodeAttribute(id, "image", undefined);
      }
    }
    requestFaces(
      order.slice(0, MAX_TILES).map((o) => o.id),
      (id) => photosFor(Number(id))?.[0],
      (id, url) => {
        setFaceImage(graph, id, url);
        scheduleFaceRepaint();
      },
    );
  }

  // One repaint per frame at most: 1,700 faces arriving would otherwise ask
  // for 1,700 refreshes.
  let repaintQueued = false;
  function scheduleFaceRepaint() {
    if (repaintQueued) return;
    repaintQueued = true;
    requestAnimationFrame(() => {
      repaintQueued = false;
      renderer.refresh({ skipIndexation: true });
    });
  }

  /** Fetch these images now, ahead of anything else in flight. */
  function preload(urls: string[]): Promise<unknown> {
    return Promise.all(urls.map((url) => new Promise((done) => {
      const img = new Image();
      // Not universally supported; where it is, it puts these ahead of the tile requests
      // already queued rather than merely alongside them.
      (img as HTMLImageElement & { fetchPriority?: string }).fetchPriority = "high";
      img.onload = img.onerror = () => done(null);
      img.src = url;
    })));
  }


  const selection: { selectedId: number | null; hoveredId: number | null } = {
    selectedId: null,
    hoveredId: null,
  };

  // Reassigned by loadLevel() on every slider change; always assigned before any handler
  // that reads them can actually run (loadLevel completes once before bootstrap()
  // returns, and nothing before that is interactive).
  let data!: GraphData;
  let graph!: ReturnType<typeof buildGraphology>;
  let renderer!: Sigma;

  // Reflects current state (fame level + selection) into the URL via replaceState -- no
  // new history entry per click, but the address bar always has a link a user can copy
  // to share exactly what they're looking at.
  function updateUrl() {
    const params = new URLSearchParams();
    if (selection.selectedId !== null) {
      const name = data.nodes.find((n) => n.id === selection.selectedId)?.name;
      if (name) params.set("person", name);
    }
    // A comparison has to survive being copied out of the address bar.
    if (openPair) {
      const a = data.nodes.find((n) => n.id === openPair!.a)?.name;
      const b = data.nodes.find((n) => n.id === openPair!.b)?.name;
      if (a && b) {
        params.set("person", a);
        params.set("vs", b);
      }
    }
    const query = params.toString();
    history.replaceState(null, "", query ? `?${query}` : location.pathname);
  }

  /** An <img> cropped to a person's face, or a lettered placeholder. */
  /** The link that reproduces what someone is looking at right now. */
  function shareLink(): string {
    const nameOf = (id: number) => data.nodes.find((n) => n.id === id)?.name;
    const params = new URLSearchParams();
    const subject = openPair ? openPair.a : selection.selectedId;
    if (subject === null) return location.origin + location.pathname;
    const name = nameOf(subject);
    if (!name) return location.origin + location.pathname;
    params.set("person", name);
    if (openPair) {
      const other = nameOf(openPair.b);
      if (other) params.set("vs", other);
    }
    return `${location.origin}${location.pathname}?${params}`;
  }

  /** Hands the link to the OS share sheet, or copies it if there isn't one. */
  async function shareCurrentView(button: HTMLButtonElement) {
    const url = shareLink();
    const title = shareTitle();
    if (navigator.share) {
      try {
        await navigator.share({ title, text: title, url });
        return;
      } catch (err) {
        if ((err as DOMException)?.name === "AbortError") return;
        // Anything else (no handler, permission denied): fall through to copy.
      }
    }
    await copyShareLink(button, url);
  }

  /** What the share sheet announces, so the preview is not a bare URL. */
  function shareTitle(): string {
    const nameOf = (id: number) => data.nodes.find((n) => n.id === id)?.name;
    if (openPair) {
      const a = nameOf(openPair.a);
      const b = nameOf(openPair.b);
      if (a && b) return `${a} and ${b} on Doppelmap`;
    }
    const name = selection.selectedId === null ? null : nameOf(selection.selectedId);
    return name ? `${name} on Doppelmap` : "Doppelmap";
  }

  /** Clipboard fallback, reporting back on the button that was pressed. */
  async function copyShareLink(button: HTMLButtonElement, link = shareLink()) {
    let ok = true;
    try {
      await navigator.clipboard.writeText(link);
    } catch {
      const field = document.createElement("textarea");
      field.value = link;
      field.setAttribute("readonly", "");
      field.style.cssText = "position:fixed;opacity:0";
      document.body.appendChild(field);
      field.select();
      try {
        ok = document.execCommand("copy");
      } catch {
        ok = false;
      }
      field.remove();
    }
    const previous = button.dataset.label ?? button.textContent ?? "";
    button.dataset.label = previous;
    button.textContent = ok ? "Copied" : "Press Ctrl+C";
    button.classList.add("copied");
    window.setTimeout(() => {
      button.textContent = button.dataset.label ?? previous;
      button.classList.remove("copied");
    }, 1600);
  }

  /** Shares the comparison as a picture. */
  async function sharePairCard(
    button: HTMLButtonElement, a: GraphNode, b: GraphNode, weight: number
  ) {
    const photoA = photosFor(a.id)?.[openPairPhotos.a];
    const photoB = photosFor(b.id)?.[openPairPhotos.b];
    if (!photoA || !photoB) return copyShareLink(button);

    const previous = button.textContent ?? "Share";
    button.disabled = true;
    button.textContent = "Making…";
    let blob: Blob;
    try {
      blob = await renderShareCard({
        a: { name: a.name, photo: photoA },
        b: { name: b.name, photo: photoB },
        percent: formatSimilarity(weight),
        band: resemblanceBand(weight)?.label ?? "similarity",
        dark: true,
      });
    } catch {
      button.disabled = false;
      button.textContent = previous;
      return copyShareLink(button);
    }
    button.disabled = false;
    button.textContent = previous;

    const name = `${a.name} and ${b.name}`.replace(/[^\w ]+/g, "").slice(0, 60);
    const file = new File([blob], `${name || "doppelmap"}.png`, { type: "image/png" });
    const url = a.id < 0 || b.id < 0 ? undefined : shareLink();
    if (navigator.canShare?.({ files: [file] })) {
      try {
        await navigator.share({ files: [file], title: `${name} on Doppelmap`, url });
        return;
      } catch (err) {
        if ((err as DOMException)?.name === "AbortError") return;
      }
    }
    // Sharing was declined or unsupported: offer the file as a download.
    const href = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = href;
    link.download = file.name;
    link.click();
    window.setTimeout(() => URL.revokeObjectURL(href), 10_000);
    button.dataset.label = previous;
    button.textContent = "Saved";
    button.classList.add("copied");
    window.setTimeout(() => {
      button.textContent = button.dataset.label ?? previous;
      button.classList.remove("copied");
    }, 1600);
  }

  /** Photo credit, behind a small info mark. */
  function renderSidebar() {
    if (selection.selectedId === null) {
      sidebarEl.hidden = true;
      sidebarEl.innerHTML = "";
      updateUrl();
      return;
    }
  /** The similar list, grouped under plain-language headings. */
  const renderSimilar = (info: ReturnType<typeof getSidebarData>) => {
    const row = (s: SidebarRow) => `
      <li data-id="${s.id}">
        ${faceHtml(s.id, s.name, "row-thumb", 36)}
        <span class="row-text">
          <span class="row-name">${escapeHtml(s.name)}</span>
          ${s.relation
            ? `<span class="row-relation">${escapeHtml(s.relation)}</span>` : ""}
        </span>
        <span class="row-percent">${s.percent}</span>
        <button class="row-compare" type="button" data-compare="${s.id}"
                data-w="${s.weight}" data-mine="${s.myPhoto}" data-theirs="${s.theirPhoto}"
                title="Compare the two closest photos"
                aria-label="Compare with ${escapeHtml(s.name)}"
          ><span aria-hidden="true">⇄</span
          ><span class="row-compare-label">Compare</span></button>
      </li>`;

    const family = info.similar.filter((s) => s.relation).length;
    const shown = hideRelatives
      ? info.similar.filter((s) => !s.relation)
      : info.similar;
    const groups = groupByResemblance(shown);
    if (!groups.length) {
      const closest = shown[0];
      const because = hideRelatives && family
        ? ` Their closest matches are family, which is hidden.`
        : "";
      return `<p class="no-resemblance">No one in this dataset looks much like
        ${escapeHtml(info.name)}.${because}${closest
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
      <button id="sidebar-dismiss" type="button" aria-label="Back to the map">
        <span class="grabber"></span>
      </button>
      <div class="sidebar-photo-wrap">
        <div id="sidebar-photo">${faceHtml(info.id, info.name, "face", 160, true)}</div>
        ${photo ? creditHtml(photo) : ""}
      </div>
      <div class="sidebar-id">
        <h2>${escapeHtml(info.name)}</h2>
        ${info.id < 0 ? "" : `<button class="share-btn" id="share-person" type="button"
                aria-label="Share ${escapeHtml(info.name)}"
              ><span class="share-icon" aria-hidden="true">${SHARE_GLYPH}</span>Share</button>`}
      </div>
      <p class="attr"></p>
      <p class="known-for" hidden></p>
      ${renderSimilar(info)}
    `;

    // A tap has no hover, so the mark toggles as well.
    for (const mark of sidebarEl.querySelectorAll<HTMLButtonElement>(".credit-btn")) {
      mark.addEventListener("click", (evt) => {
        evt.stopPropagation();
        mark.parentElement?.classList.toggle("open");
      });
    }

    sidebarEl.querySelector<HTMLButtonElement>("#sidebar-dismiss")
      ?.addEventListener("click", deselectManually);

    sidebarEl.querySelector<HTMLButtonElement>("#share-person")
      ?.addEventListener("click", (evt) => shareCurrentView(evt.currentTarget as HTMLButtonElement));

    sidebarEl.querySelectorAll<HTMLLIElement>("li[data-id]").forEach((li) => {
      li.addEventListener("click", () => selectNodeManually(Number(li.dataset.id)));

      // Enlarged preview of whatever this row already shows -- no extra request, since
      // it is the same Commons image at a bigger width.
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
        // Without this the row's own click handler also fires and navigates away from
        // the person we are trying to compare against.
        evt.stopPropagation();
        hoverPreview.hidden = true;
        if (selection.selectedId === null) return;
        const me = selection.selectedId;
        const buttons = [...sidebarEl.querySelectorAll<HTMLButtonElement>("button[data-compare]")];
        const items: PairRef[] = buttons.map((b) => ({
          a: me, b: Number(b.dataset.compare), weight: Number(b.dataset.w),
          photoA: Number(b.dataset.mine), photoB: Number(b.dataset.theirs),
        }));
        showPairFrom(items, buttons.indexOf(compare));
      });
    });

    updateUrl();

    // Wikipedia supplies the one-line description and a link out.
    if (info.id < 0) return;

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

  /** Centre of the canvas area not covered by UI, in viewport pixels. */
  function visibleCentre(): { x: number; y: number } {
    const { width, height } = renderer.getDimensions();
    const top = toolbarEl.getBoundingClientRect().bottom;
    if (sidebarEl.hidden) return { x: width / 2, y: (top + height) / 2 };

    const sidebar = sidebarEl.getBoundingClientRect();
    // A bottom sheet spans the full width; the desktop sidebar is docked right and
    // leaves the canvas to its left.
    if (sidebar.width >= width * 0.9) {
      return { x: width / 2, y: (top + sidebar.top) / 2 };
    }
    return { x: sidebar.left / 2, y: (top + height) / 2 };
  }

  function selectNode(id: number) {
    selection.selectedId = id;
    resultsEl.innerHTML = "";
    // Sidebar first: visibleCentre() measures it, so it has to be on screen and laid out
    // before the camera target is computed.
    renderSidebar();
    flyToNode(renderer, String(id), visibleCentre());
    // nodeReducer/edgeReducer output is cached and only re-evaluated on refresh() -- the
    // mouse hovering the graph triggers that incidentally (enterNode/leaveNode both call
    // it), which is why a manual click happened to look right, but any selection made
    // without the mouse over the destination node (search, sidebar and dashboard clicks)
    // left the highlight frozen on whatever it was before.
    renderer.refresh();
  }

  function selectNodeManually(id: number) {
    selectNode(id);
  }

  // On a phone the panel is a bottom sheet covering more than half the screen, and the
  // only way back to the map was to find and tap a node behind it.
  let touchStartY: number | null = null;
  sidebarEl.addEventListener("touchstart", (evt) => {
    touchStartY = sidebarEl.scrollTop <= 0 ? evt.touches[0].clientY : null;
  }, { passive: true });
  // Not passive, deliberately.
  sidebarEl.addEventListener("touchmove", (evt) => {
    if (touchStartY === null) return;
    const moved = evt.touches[0].clientY - touchStartY;
    if (moved > 0) {
      evt.preventDefault();
      sidebarEl.style.transform = `translateY(${moved}px)`;
    } else {
      sidebarEl.style.transform = "";
    }
  }, { passive: false });
  sidebarEl.addEventListener("touchend", (evt) => {
    const moved = touchStartY === null
      ? 0
      : (evt.changedTouches[0]?.clientY ?? touchStartY) - touchStartY;
    touchStartY = null;
    sidebarEl.style.transform = "";
    if (moved > 90) deselectManually();
  }, { passive: true });

  function deselectManually() {
    selection.selectedId = null;
    resultsEl.innerHTML = "";
    renderSidebar();
    renderer.refresh();
  }

  function pickRandomNodeId(): number {
    return data.nodes[Math.floor(Math.random() * data.nodes.length)].id;
  }

  // The whole point of the site is "these two look alike", but until now you could only
  // ever see one face at a time -- a pair was rendered as one photo plus the other
  // person's name as text.
  /** Opens a comparison, optionally as position `index` of a ranked list. */
  function showPairFrom(items: PairRef[], index: number) {
    const clamped = Math.min(Math.max(index, 0), items.length - 1);
    const pair = items[clamped];
    if (!pair) return;
    pairSeries = { items, index: clamped };
    showPair(pair.a, pair.b, pair.weight, pair.photoA, pair.photoB, true);
  }

  function renderPairNav() {
    const series = pairSeries;
    pairNav.hidden = !series || series.items.length < 2;
    if (!series) return;
    pairPrev.disabled = series.index <= 0;
    pairNext.disabled = series.index >= series.items.length - 1;
    pairPosition.textContent = `${series.index + 1} of ${series.items.length}`;
  }

  function stepPair(delta: number) {
    if (pairSeries) showPairFrom(pairSeries.items, pairSeries.index + delta);
  }

  function showPair(
    idA: number, idB: number, weight: number, photoA = 0, photoB = 0,
    keepSeries = false
  ) {
    // Opened on its own (a node click, a shared link): it belongs to no list, so the
    // pager goes away rather than paging through something stale.
    if (!keepSeries) pairSeries = null;
    const nodesById = new Map(data.nodes.map((n) => [n.id, n]));
    const a = nodesById.get(idA);
    const b = nodesById.get(idB);
    if (!a || !b) return;

    // The specific photographs the model matched, chosen at build time -- not two
    // arbitrary portraits.
    const side = (n: GraphNode, which: number) => {
      const photo: PhotoRef | undefined = photosFor(n.id)?.[which] ?? photosFor(n.id)?.[0];
      const media = photo
        ? `<span class="pair-face" role="img" aria-label="${escapeHtml(n.name)}"
                 style="${faceCropStyle(photo, 0.75, 190)}"></span>`
        : `<span class="face-fallback pair-fallback">${escapeHtml(
            n.name.split(" ").map((w) => w[0]).join("").slice(0, 2))}</span>`;
      const credit = photo ? creditHtml(photo, "pair-credit") : "";
      return `
        <figure class="pair-side">
          ${media}
          <figcaption>${escapeHtml(n.name)}</figcaption>
          ${credit}
          <button class="pair-goto" type="button" data-goto="${n.id}">View on map</button>
        </figure>`;
    };

    // The same wording the sidebar groups by, so a score means one thing everywhere it
    // appears.
    const band = resemblanceBand(weight);

    const blendable = photosFor(a.id)?.[photoA] && photosFor(b.id)?.[photoB];
    const blend = blendable
      ? `<div class="pair-blend auto">
           <div class="blend-stack">
             <span class="blend-face" style="${faceCropStyle(
               photosFor(a.id)![photoA] ?? photosFor(a.id)![0], 0.75, 150)}"></span>
             <span class="blend-face blend-top" id="blend-top" style="${faceCropStyle(
               photosFor(b.id)![photoB] ?? photosFor(b.id)![0], 0.75, 150)}"></span>
           </div>
           <div class="blend-caption"><span>blend</span></div>
         </div>`
      : "";

    pairBody.innerHTML = `
      ${band
        ? `<div class="pair-verdict" data-band="${band.slug}">${escapeHtml(band.label)}</div>`
        : ""}
      ${side(a, photoA)}
      <div class="pair-score">
        <strong>${formatSimilarity(weight)}</strong>
        ${band ? "" : `<span>similarity</span>`}
        <button class="share-btn" id="share-pair" type="button"
                aria-label="Share this comparison"
        ><span class="share-icon" aria-hidden="true">${SHARE_GLYPH}</span>Share</button>
      </div>
      <div class="pair-scale">${escapeHtml(rarityNote(weight, data.meta.scale))}</div>
      ${blend}
      ${a.qid && b.qid ? `
      <div class="pair-vote" id="pair-vote">
        <span class="ask">Do they look alike?</span>
        <span class="buttons">
          <button type="button" data-vote="yes" aria-label="Yes, they look alike">&#10003; Yes</button>
          <button type="button" data-vote="no" aria-label="No, they do not look alike">&#10007; No</button>
        </span>
        <span class="tally" id="pair-tally"></span>
      </div>` : ""}
      ${side(b, photoB)}
    `;
    openPair = { a: idA, b: idB };
    updateUrl();
    openPairPhotos = { a: photoA, b: photoB };
    renderPairNav();
    pairView.hidden = false;

    for (const mark of pairBody.querySelectorAll<HTMLButtonElement>(".credit-btn")) {
      mark.addEventListener("click", (evt) => {
        evt.stopPropagation();
        mark.parentElement?.classList.toggle("open");
      });
    }

    wireVoting(a, b);

    pairBody.querySelector<HTMLButtonElement>("#share-pair")
      ?.addEventListener("click", (evt) =>
        sharePairCard(evt.currentTarget as HTMLButtonElement, a, b, weight));

    pairBody.querySelectorAll<HTMLButtonElement>("button[data-goto]").forEach((btn) => {
      btn.addEventListener("click", () => {
        closePair();
        selectNodeManually(Number(btn.dataset.goto));
      });
    });
  }

  /** The "do they look alike?" row under a comparison. */
  function wireVoting(a: GraphNode, b: GraphNode) {
    const row = pairBody.querySelector<HTMLElement>("#pair-vote");
    if (!row || !a.qid || !b.qid) return;
    const tallyEl = row.querySelector<HTMLElement>("#pair-tally");
    const buttons = [...row.querySelectorAll<HTMLButtonElement>("button[data-vote]")];

    const paint = (mine: boolean | undefined, text: string) => {
      for (const button of buttons) {
        button.classList.toggle("chosen", mine !== undefined
          && (button.dataset.vote === "yes") === mine);
      }
      if (tallyEl) tallyEl.textContent = text;
    };

    // Already answered this pair in this tab: show where it stands.
    const existing = myVote(a.qid, b.qid);
    if (existing !== undefined) {
      paint(existing, "");
      void pairTally(a.qid, b.qid).then((tally) => {
        if (tally && openPair?.a === a.id && openPair?.b === b.id) {
          paint(existing, describeTally(tally));
        }
      });
    }

    for (const button of buttons) {
      button.addEventListener("click", async () => {
        const yes = button.dataset.vote === "yes";
        const qidA = a.qid as string;
        const qidB = b.qid as string;
        for (const other of buttons) other.disabled = true;
        paint(yes, "Counting...");
        const tally = await castVote(qidA, qidB, yes);
        for (const other of buttons) other.disabled = false;
        // The card may have been paged past while the request was in flight.
        if (openPair?.a !== a.id || openPair?.b !== b.id) return;
        if (!tally) {
          paint(undefined, "Could not record that vote.");
          return;
        }
        rememberVote(qidA, qidB, yes);
        paint(yes, describeTally(tally));
      });
    }
  }

  function closePair() {
    openPair = null;   // so Share falls back to the person again
    updateUrl();       // and the address bar drops ?vs= with it
    pairSeries = null;
    pairNav.hidden = true;
    pairView.hidden = true;
    pairBody.innerHTML = "";
  }

  // How many pairs the Top Pairs panel lists.
  const TOP_PAIRS_SHOWN = 100;

  /** Is this edge a family resemblance? */
  function edgeIsFamily(a: number, b: number): boolean {
    const row = (data.similar[String(a)] ?? []).find((e) => e[0] === b);
    return Boolean(row && row.length > 4);
  }

  function renderDashboard() {
    const nodesById = new Map(data.nodes.map((n) => [n.id, n]));
    // The ranking is where family dominates most: these are the strongest resemblances
    // in the whole map, and 29% of those are relatives.
    const topPairs = [...data.edges]
      .filter(([a, b]) => !(hideRelatives && edgeIsFamily(a, b)))
      .sort((a, b) => b[2] - a[2])
      .slice(0, TOP_PAIRS_SHOWN);
    dashboardListEl.innerHTML = topPairs
      .map(([a, b, w]) => {
        const nameA = nodesById.get(a)?.name ?? "Unknown";
        const nameB = nodesById.get(b)?.name ?? "Unknown";
        const match = (data.similar[String(a)] ?? []).find((e) => e[0] === b);
        // Two faces beside the names: the list is a ranking of resemblances, and a
        // resemblance is not a thing two names can show you.
        return `<li data-a="${a}" data-b="${b}" data-w="${w}"
                    data-mine="${match?.[2] ?? 0}" data-theirs="${match?.[3] ?? 0}">
          <span class="pair-row-faces">
            ${faceHtml(a, nameA, "row-thumb", 30)}${faceHtml(b, nameB, "row-thumb", 30)}
          </span>
          <span class="pair-row-names">${escapeHtml(nameA)} ↔ ${escapeHtml(nameB)}</span>
          <span class="pair-row-score">${formatSimilarity(w)}</span>
        </li>`;
      })
      .join("");
    dashboardListEl.querySelectorAll<HTMLLIElement>("li[data-a]").forEach((li) => {
      // The same enlarged look the sidebar rows have, but two faces wide: a Top Pairs
      // row claims a resemblance, and one face cannot show it.
      li.addEventListener("mouseenter", () => {
        const a = photosFor(Number(li.dataset.a))?.[Number(li.dataset.mine)];
        const b = photosFor(Number(li.dataset.b))?.[Number(li.dataset.theirs)];
        if (!a || !b) return;
        const faces = hoverPair.querySelectorAll<HTMLElement>(".hover-face");
        faces[0].setAttribute("style", faceCropStyle(a, 0.55, 150));
        faces[1].setAttribute("style", faceCropStyle(b, 0.55, 150));

        // Left of the panel: the sidebar sits immediately to its right, so there is no
        // room on that side.
        hoverPair.hidden = false;
        const panel = dashboardEl.getBoundingClientRect();
        const own = hoverPair.getBoundingClientRect();
        const row = li.getBoundingClientRect();
        hoverPair.style.left = `${Math.max(8, panel.left - own.width - 12)}px`;
        hoverPair.style.top =
          `${Math.min(window.innerHeight - own.height - 8,
                      Math.max(8, row.top + row.height / 2 - own.height / 2))}px`;
      });
      li.addEventListener("mouseleave", () => {
        hoverPair.hidden = true;
      });

      li.addEventListener("click", () => {
        const rows = [...dashboardListEl.querySelectorAll<HTMLLIElement>("li[data-a]")];
        const items: PairRef[] = rows.map((row) => ({
          a: Number(row.dataset.a), b: Number(row.dataset.b),
          weight: Number(row.dataset.w),
          photoA: Number(row.dataset.mine), photoB: Number(row.dataset.theirs),
        }));
        showPairFrom(items, rows.indexOf(li));
        hoverPair.hidden = true;
        dashboardEl.hidden = true;
      });
    });
  }

  async function loadGraph() {
    data = await loadGraphData(`${import.meta.env.BASE_URL}data/${GRAPH_FILE}`);
    // Kicked off, not awaited: photos.json is several times the size of the graph and
    // only needed once someone opens a person, so the map appears without waiting for
    // it.
    photosReady = loadPhotos(`${import.meta.env.BASE_URL}data/${PHOTOS_FILE}`)
      .then(() => {
        if (selection.selectedId !== null) renderSidebar();
        // The Top Pairs list is built before the photos arrive, so its faces would stay
        // as initials for the life of the page without this.
        renderDashboard();
      });

    // Faces on the nodes, unless the tiles do not belong to this build.
    showFaces = new URLSearchParams(location.search).get("faces") !== "0";
    graph = buildGraphology(data, showFaces);
    renderer = new Sigma(graph, container as HTMLElement, {
      // Our nodes are deliberately small (1.5-4px), so Sigma's default size threshold
      // would suppress every label.
      ...(showFaces
        ? {
            defaultNodeType: "image",
            nodeProgramClasses: {
              image: createNodeImageProgram({
                keepWithinCircle: true,
                // Matches the face tile size. The atlas holds every face ever
                // drawn and is never trimmed, so the tile is kept small: at
                // 72px the whole collection is ~35 MB of texture.
                size: { mode: "force", value: 72 },
                maxTextureSize: 2048,
              }),
            },
          }
        : {}),
      labelRenderedSizeThreshold: 0,
      labelGridCellSize: 250,
      labelDensity: 0.6,
      labelColor: { color: labelColor() },
      defaultDrawNodeLabel: drawNodeLabel,
      defaultDrawNodeHover: drawNodeHover,
      // Stop the map shrinking into a speck in the middle of empty space.
      maxCameraRatio: MAX_ZOOM_OUT_RATIO,
      // Zooming in was unbounded, so a node could be drawn at any size while its face
      // texture stays 128px -- the picture got bigger and softer and never sharper, and
      // past a point there was nothing on screen but edge bands.
      minCameraRatio: MAX_ZOOM_IN_RATIO,
      enableCameraRotation: false,
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
      const ratio = renderer.getCamera().ratio;
      const mode = getDisplayMode(ratio);
      if (mode === "dot") display.label = "";
      if (ratio > FACE_RATIO) display.image = undefined;

      // A click "sticks" the highlight even after the mouse moves away; hovering a
      // (different) node temporarily previews its neighbors on top of that.
      const highlightId = selection.hoveredId ?? selection.selectedId;
      if (highlightId !== null) {
        const highlightKey = String(highlightId);
        const isHighlighted = nodeId === highlightKey;
        const isNeighbor = graph.areNeighbors(nodeId, highlightKey);
        if (!isHighlighted && !isNeighbor) {
          display.color = DIM_NODE_COLOR;
        }
      }

      // A face cannot be dimmed -- the photograph covers whatever colour is
      // under it -- so everyone outside a selection drops back to a dot,
      // leaving the chosen person and their matches as the only faces. Only
      // for a deliberate selection: doing it on hover wipes the map as the
      // pointer crosses it.
      const focusKey = selection.selectedId === null ? null : String(selection.selectedId);
      if (focusKey !== null && nodeId !== focusKey && !graph.areNeighbors(nodeId, focusKey)) {
        display.color = DIM_NODE_COLOR;
        display.image = undefined;
        display.label = "";
      }

      // The selected node itself gets its own look, independent of hover -- otherwise
      // it's only as visible as "not dimmed", easy to lose track of at a glance,
      // especially at low degree (small size).
      if (selection.selectedId !== null && nodeId === String(selection.selectedId)) {
        display.color = SELECTED_NODE_COLOR;
        display.size = attrs.size + 2;
        display.zIndex = 1;
      }
      return display;
    });

    renderer.setSetting("edgeReducer", (edge, attrs) => {
      const display = { ...attrs };
      // Sigma scales edge width with the zoom, so a link that reads as a line
      // on the whole map becomes a band once you are among the faces.
      const ratio = renderer.getCamera().ratio;
      if (ratio < 1) display.size = (attrs.size as number) * Math.max(0.3, ratio ** 0.6);
      const highlightId = selection.hoveredId ?? selection.selectedId;
      if (highlightId !== null) {
        const highlightKey = String(highlightId);
        const extremities = graph.extremities(edge);
        if (!extremities.includes(highlightKey)) {
          // Edges away from the highlighted node are dimmed, not hidden.
          display.color = fadedEdgeColor();
        }
      }
      return display;
    });

    // nodeReducer's output is cached and only re-runs on refresh() (which camera
    // pan/zoom does NOT trigger on its own), so without this the zoom-dependent label
    // mode would only update by coincidence, e.g.
    let lastDisplayMode = getDisplayMode(renderer.getCamera().ratio);
    // Panning and zooming bring new people into view, and each needs its photograph
    // fetched and cropped.
    let faceSweep = 0;
    const scheduleFaceSweep = () => {
      if (faceSweep) return;
      faceSweep = window.setTimeout(() => {
        faceSweep = 0;
        cutVisibleFaces();
      }, 180);
    };

    let clamping = false;
    renderer.getCamera().on("updated", () => {
      scheduleFaceSweep();
      const camera = renderer.getCamera();
      const mode = getDisplayMode(camera.ratio);
      if (mode !== lastDisplayMode) {
        lastDisplayMode = mode;
        renderer.refresh({ skipIndexation: true });
      }

      // Panning had no limit, so the map could be dragged away entirely, leaving an
      // empty background with no way back but a reload.
      if (clamping) return;
      // Around the centre, not the bounding box.
      const reach = PAN_REACH + PAN_MARGIN * camera.ratio;
      const clamp = (v: number) =>
        Math.min(0.5 + reach, Math.max(0.5 - reach, v));
      const x = clamp(camera.x);
      const y = clamp(camera.y);
      if (x !== camera.x || y !== camera.y) {
        clamping = true;
        camera.setState({ x, y });
        clamping = false;
      }
    });

    renderDashboard();
  }

  window.addEventListener("keydown", (evt) => {
    if (evt.key !== "Escape") return;
    // Escape closes the topmost thing first rather than always clearing the selection
    // underneath the pair view.
    if (!pairView.hidden) {
      closePair();
    } else {
      deselectManually();
    }
  });



  pairClose.addEventListener("click", closePair);
  pairExplore.addEventListener("click", closePair);
  document.addEventListener("keydown", (evt) => {
    if (evt.key !== "Escape") return;
    if (!aboutView.hidden) aboutView.hidden = true;
    else if (!pairView.hidden) closePair();
  });

  pairPrev.addEventListener("click", () => stepPair(-1));
  pairNext.addEventListener("click", () => stepPair(1));
  // Arrow keys page the comparison, which is what a pager invites you to try.
  document.addEventListener("keydown", (evt) => {
    if (pairView.hidden || !pairSeries) return;
    if (evt.key === "ArrowLeft") stepPair(-1);
    if (evt.key === "ArrowRight") stepPair(1);
  });
  pairView.addEventListener("click", (evt) => {
    // Backdrop only -- clicks inside the card shouldn't dismiss it.
    if (evt.target === pairView) closePair();
  });

  // A standing preference, so it lives in the toolbar rather than inside one person's
  // panel: there it was invisible until you happened to open someone with a relative,
  // which is 4% of people.
  const hideRelativesEl = document.getElementById("hide-relatives") as HTMLInputElement | null;
  if (hideRelativesEl) {
    hideRelativesEl.checked = hideRelatives;
    hideRelativesEl.addEventListener("change", () => {
      hideRelatives = hideRelativesEl.checked;
      try {
        localStorage.setItem("doppelmap.hideRelatives", hideRelatives ? "1" : "0");
      } catch {
        // Not being able to remember it is not a reason to ignore it.
      }
      if (selection.selectedId !== null) renderSidebar();
      renderDashboard();
    });
  }

  // About is a dialog over the whole page now, not a dropdown hanging off the toolbar:
  // it is a page of prose, and reading it is the only thing you are doing while it is
  // open.
  aboutToggle.addEventListener("click", () => {
    aboutView.hidden = false;
  });
  aboutClose.addEventListener("click", () => {
    aboutView.hidden = true;
  });
  aboutView.addEventListener("click", (evt) => {
    if (evt.target === aboutView) aboutView.hidden = true;
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
  // Deferred one frame: right after `new Sigma(...)`, in the same tick, its display data
  // (node positions in camera space) hasn't been computed by an actual render pass yet,
  // so flyToNode's camera.animate() targets stale/default coordinates instead of the
  // node's real position -- the camera visibly moves but doesn't end up centered on
  // anyone.
  requestAnimationFrame(() => {
    if (initialPersonName) {
      const match = data.nodes.find((n) => n.name === initialPersonName);
      if (match) {
        selectNode(match.id);
        // A ?vs= link should land on the comparison itself.
        if (initialVersusName) {
          const other = data.nodes.find((n) => n.name === initialVersusName);
          const entry = other
            ? (data.similar[String(match.id)] ?? []).find(([id]) => id === other.id)
            : undefined;
          if (other && entry) {
            // After the photos, not before.
            photosReady.then(() => {
              showPair(match.id, other.id, entry[1], entry[2] ?? 0, entry[3] ?? 0);
              enableFaces();
            }, enableFaces);
          } else {
            photosReady.then(enableFaces, enableFaces);
          }
        } else {
          photosReady.then(enableFaces, enableFaces);
        }
        return;
      }
    }
    // Land on a worked example rather than an empty map, but only once at startup.
      const landing = data.landing ?? [];
    if (!landing.length) {
      selectNode(pickRandomNodeId());
      photosReady.then(enableFaces, enableFaces);
      return;
    }

    // The whole curated set, as something to page through rather than a single pair and
    // a dead end.
    const items: PairRef[] = landing
      .map((id) => ({ id, best: (data.similar[String(id)] ?? [])[0] }))
      .filter((x) => x.best)
      .map((x) => ({
        a: x.id, b: x.best[0], weight: x.best[1],
        photoA: x.best[2] ?? 0, photoB: x.best[3] ?? 0,
      }))
      .sort(() => Math.random() - 0.5);

    selectNode(items[0].a);
    // After the photos, not before: the comparison is two faces, and photos.json is
    // fetched lazily.
    photosReady
      .then(() => {
        const first = items[0];
        const a = photosFor(first.a)?.[first.photoA] ?? photosFor(first.a)?.[0];
        const b = photosFor(first.b)?.[first.photoB] ?? photosFor(first.b)?.[0];
        const urls = [a, b].filter(Boolean).map((p) => photoUrl(p!, 640));
        // Two requests, and the card is drawn either way -- a Commons outage should not
        // leave a visitor staring at the map's loading state with no comparison at all.
        return Promise.race([preload(urls), new Promise((r) => setTimeout(r, 4000))]);
      })
      .then(() => {
        showPairFrom(items, 0);
        // Only now: the two photographs on screen are in, so the tiles behind them can
        // have the connection.
        enableFaces();
      })
      .catch(enableFaces);
  });


}

bootstrap().catch((err) => {
  console.error(err);
});
