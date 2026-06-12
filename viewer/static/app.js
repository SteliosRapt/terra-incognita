/* app.js — polls /api/state, draws the world, runs the tabbed console. */
"use strict";

const SPR = buildSprites();
const TILE = 16;
const POLL_MS = 700;

let state = null;        // latest /api/state payload
let selected = null;     // selected agent id
let waterFrame = 0, fireFrame = 0;
let IDX = {};            // live run: agent id -> color/sprite index

const $ = (id) => document.getElementById(id);
const map = $("map"), ctx = map.getContext("2d");
ctx.imageSmoothingEnabled = false;

/* persisted UI state */
const ui = {
  tab: localStorage.getItem("ti.tab") || "overview",
  sideW: +localStorage.getItem("ti.sideW") || 420,
  trails: localStorage.getItem("ti.trails") === "1",
  heat: localStorage.getItem("ti.heat") === "1",
  evCat: "all",
  evAgents: null,          // null = all; otherwise Set of ids
};

/* ------------------------------------------------------------- helpers */
function colorIdx(id) {
  if (id in IDX) return IDX[id];
  const i = parseInt(String(id).slice(1), 10);
  return Number.isFinite(i) ? i % AGENT_COLORS.length : 0;
}
function colorOf(id) { return AGENT_COLORS[colorIdx(id) % AGENT_COLORS.length][0]; }
function spriteOf(id) { return SPR.agents[colorIdx(id) % SPR.agents.length]; }

async function post(url, body) {
  const r = await fetch(url, { method: "POST", body: JSON.stringify(body || {}) });
  const data = await r.json().catch(() => ({}));
  if (!r.ok || data.error) showError(data.error || `${url} failed (${r.status})`);
  return data;
}
function showError(msg) {
  const b = $("errorBanner");
  b.textContent = msg;
  b.classList.remove("hidden");
  setTimeout(() => b.classList.add("hidden"), 6000);
}
function esc(s) {
  const d = document.createElement("span");
  d.textContent = s == null ? "" : String(s);
  return d.innerHTML;
}

/* Update an element's HTML only when it actually changed, preserving scroll
   position and which <details data-k=…> were open — so polling never yanks
   the UI out from under the user. Returns true if the DOM was touched. */
function setHTML(el, html) {
  if (el._lastHTML === html) return false;
  const open = new Set(Array.from(el.querySelectorAll("details[data-k]"))
    .filter((d) => d.open).map((d) => d.dataset.k));
  const st = el.scrollTop;
  el.innerHTML = html;
  for (const d of el.querySelectorAll("details[data-k]"))
    if (open.has(d.dataset.k)) d.open = true;
  el.scrollTop = st;
  el._lastHTML = html;
  return true;
}

function fmtDur(sec) {
  if (!isFinite(sec) || sec < 0) return "—";
  if (sec < 90) return Math.round(sec) + "s";
  if (sec < 5400) return Math.round(sec / 60) + "m";
  return Math.floor(sec / 3600) + "h " + String(Math.round(sec % 3600 / 60)).padStart(2, "0") + "m";
}
function fmtDate(epoch) {
  const d = new Date(epoch * 1000);
  return d.toLocaleDateString(undefined, { month: "short", day: "numeric" }) +
    " " + d.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" });
}

/* portraits + item icons: stamped into placeholder canvases after setHTML */
function blitArt(root) {
  for (const c of root.querySelectorAll("canvas.portrait:not([data-done])")) {
    c.width = 16; c.height = 16;
    const g = c.getContext("2d");
    g.imageSmoothingEnabled = false;
    g.fillStyle = "#1a2310";
    g.fillRect(0, 0, 16, 16);
    g.drawImage(SPR.agents[+c.dataset.i % SPR.agents.length], 0, 0);
    c.dataset.done = "1";
  }
  for (const c of root.querySelectorAll("canvas[data-item]:not([data-done])")) {
    c.width = 16; c.height = 16;
    const g = c.getContext("2d");
    g.imageSmoothingEnabled = false;
    g.drawImage(itemIcon(SPR, c.dataset.item), 0, 0);
    c.dataset.done = "1";
  }
}

/* minecraft-ish slot grid: 1 slot = up to 5 units of one item */
function slotsHTML(inventory, tools, capacity) {
  const slots = [];
  for (const [item, qty] of Object.entries(inventory).sort()) {
    let left = qty;
    while (left > 0) {
      const q = Math.min(left, 5);
      slots.push(`<span class="slot"><canvas data-item="${esc(item)}" title="${esc(item)} ×${q}"></canvas><span class="q">${q}</span></span>`);
      left -= q;
    }
  }
  for (let i = slots.length; i < capacity; i++) slots.push(`<span class="slot"></span>`);
  // tools don't occupy inventory slots — shown after the grid, gold border
  for (const t of tools || [])
    slots.push(`<span class="slot tool"><canvas data-item="${esc(t)}" title="${esc(t)} (tool)"></canvas></span>`);
  return `<div class="slots">${slots.join("")}</div>`;
}
function capacityOf(a) { return a.tools && a.tools.includes("basket") ? 16 : 8; }

/* tick-rate sampling for the ETA (client-side: works even mid-run) */
let rateSamples = [], rateRun = null;
function sampleRate(runName, status, tick) {
  if (runName !== rateRun || status !== "running") {
    if (runName !== rateRun || status === "paused") rateSamples = [];
    rateRun = runName;
    if (status !== "running") return;
  }
  const last = rateSamples[rateSamples.length - 1];
  if (!last || tick > last.tick) {
    rateSamples.push({ tick, t: Date.now() });
    if (rateSamples.length > 40) rateSamples.shift();
  }
}
function etaText(total, tick) {
  if (rateSamples.length < 3) return "";
  const a = rateSamples[0], b = rateSamples[rateSamples.length - 1];
  const sPerTick = (b.t - a.t) / 1000 / (b.tick - a.tick);
  return { rate: sPerTick, eta: fmtDur((total - tick) * sPerTick) };
}

/* ------------------------------------------------------------- tabs */
function setTab(name) {
  ui.tab = name;
  localStorage.setItem("ti.tab", name);
  for (const b of document.querySelectorAll("#tabs button"))
    b.classList.toggle("on", b.dataset.tab === name);
  for (const p of document.querySelectorAll(".pane"))
    p.classList.toggle("on", p.id === "tab-" + name);
  if (name === "files") loadRuns();
  if (name === "replay") rpFillRunSelect();
  renderPanels();
}
for (const b of document.querySelectorAll("#tabs button"))
  b.onclick = () => setTab(b.dataset.tab);

/* splitter drag → console width */
function applySideW() {
  ui.sideW = Math.max(310, Math.min(760, ui.sideW));
  document.documentElement.style.setProperty("--side-w", ui.sideW + "px");
}
applySideW();
$("splitter").addEventListener("mousedown", (e) => {
  e.preventDefault();
  const startX = e.clientX, startW = ui.sideW;
  const move = (ev) => { ui.sideW = startW + (startX - ev.clientX); applySideW(); };
  const up = () => {
    document.removeEventListener("mousemove", move);
    document.removeEventListener("mouseup", up);
    localStorage.setItem("ti.sideW", ui.sideW);
  };
  document.addEventListener("mousemove", move);
  document.addEventListener("mouseup", up);
});

/* overlay toggles */
function syncToggles() {
  $("tgTrails").classList.toggle("on", ui.trails);
  $("tgHeat").classList.toggle("on", ui.heat);
}
$("tgTrails").onclick = () => { ui.trails = !ui.trails; localStorage.setItem("ti.trails", ui.trails ? "1" : "0"); syncToggles(); };
$("tgHeat").onclick = () => { ui.heat = !ui.heat; localStorage.setItem("ti.heat", ui.heat ? "1" : "0"); syncToggles(); };
syncToggles();

/* ------------------------------------------------------------- polling */
async function poll() {
  try {
    const r = await fetch("/api/state");
    state = await r.json();
    renderPanels();
  } catch (e) { /* server briefly busy — keep last state */ }
  setTimeout(poll, POLL_MS);
}

/* ------------------------------------------------------------- replay */
const replay = {
  data: null, evByTick: null, tick: 0, playing: false, timer: null,
  st: null, stTick: -1,
};
function replayActive() { return ui.tab === "replay" && replay.st; }

function rpFillRunSelect() {
  fetch("/api/runs").then((r) => r.json()).then((runs) => {
    const cur = $("rpRun").value;
    $("rpRun").innerHTML = runs.map((r) =>
      `<option value="${esc(r.name)}">${esc(r.name)}</option>`).join("");
    if (cur && runs.some((r) => r.name === cur)) $("rpRun").value = cur;
  });
}

function rpReset() {
  const d = replay.data;
  replay.st = {
    tiles: new Map(),                       // "x,y" -> char override
    structures: new Map(),                  // "x,y" -> type
    agents: new Map(d.agents0.map((a) => [a.id, { x: a.x, y: a.y, alive: true }])),
    trails: new Map(d.agents0.map((a) => [a.id, []])),
    heat: new Map(),
  };
  replay.stTick = -1;
}

function rpApply(ev) {
  const st = replay.st, p = ev.payload || {};
  const key = ev.position ? ev.position[0] + "," + ev.position[1] : null;
  // any positioned event pins its actor there (gather-travel emits no moved
  // events, so gathered/interrupted/ate/… are how travelers reappear)
  if (ev.actor && ev.position) {
    const ag = st.agents.get(ev.actor);
    if (ag) {
      ag.x = ev.position[0]; ag.y = ev.position[1];
      if (ev.type === "moved") {
        const tr = st.trails.get(ev.actor);
        tr.push([ag.x, ag.y]);
        if (tr.length > 40) tr.shift();
        st.heat.set(key, (st.heat.get(key) || 0) + 1);
      }
    }
  }
  switch (ev.type) {
    case "resource_depleted":
      st.tiles.set(key, p.resource === "berries" ? "g" : p.resource === "wood" ? "t" : "m");
      break;
    case "resource_regrown":
      st.tiles.set(key, p.resource === "berries" ? "b" : "T");
      break;
    case "structure_built":
      st.structures.set(key, p.structure || "fire");
      break;
    case "structure_expired":
      st.structures.delete(key);
      break;
    case "agent_died": {
      const ag = st.agents.get(ev.actor);
      if (ag) { ag.alive = false; if (ev.position) { ag.x = ev.position[0]; ag.y = ev.position[1]; } }
      break;
    }
  }
}

function rpSeek(t) {
  t = Math.max(0, Math.min(t, replay.data.ticks - 1));
  if (t < replay.stTick) rpReset();
  for (let tt = replay.stTick + 1; tt <= t; tt++)
    for (const ev of replay.evByTick[tt] || []) rpApply(ev);
  replay.stTick = t;
  replay.tick = t;
  $("rpScrub").value = t;
  $("rpTick").textContent = "t" + t;
  renderReplayEvents();
}

function rpLoad(name) {
  $("rpInfo").textContent = "loading " + name + "…";
  fetch("/api/replay?run=" + encodeURIComponent(name))
    .then((r) => r.json())
    .then((d) => {
      if (d.error) { $("rpInfo").textContent = ""; showError(d.error); return; }
      replay.data = d;
      replay.evByTick = [];
      for (const ev of d.events)
        (replay.evByTick[ev.tick] = replay.evByTick[ev.tick] || []).push(ev);
      rpStop();
      rpReset();
      $("rpScrub").max = Math.max(0, d.ticks - 1);
      $("rpScrub").disabled = false;
      $("rpPlay").disabled = false;
      $("rpInfo").textContent =
        `${d.name} — ${d.agents0.length} agents · ${d.ticks} ticks · seed ${d.config.seed} · ${d.config.knowledge_condition}`;
      rpSeek(0);
    })
    .catch((e) => { $("rpInfo").textContent = ""; showError("replay load failed: " + e); });
}

function rpStop() {
  replay.playing = false;
  if (replay.timer) { clearInterval(replay.timer); replay.timer = null; }
  $("rpPlay").textContent = "▶";
}
function rpPlayPause() {
  if (!replay.data) return;
  if (replay.playing) { rpStop(); return; }
  replay.playing = true;
  $("rpPlay").textContent = "❚❚";
  const tps = +$("rpSpeed").value;
  replay.timer = setInterval(() => {
    if (replay.tick >= replay.data.ticks - 1) { rpStop(); return; }
    rpSeek(replay.tick + 1);
  }, 1000 / tps);
}
$("rpLoad").onclick = () => { if ($("rpRun").value) rpLoad($("rpRun").value); };
$("rpPlay").onclick = rpPlayPause;
$("rpScrub").oninput = (e) => { rpStop(); rpSeek(+e.target.value); };
$("rpSpeed").onchange = () => { if (replay.playing) { rpStop(); rpPlayPause(); } };

const RP_SKIP = new Set(["moved", "resource_regrown", "resource_depleted"]);
function renderReplayEvents() {
  const evs = (replay.evByTick[replay.tick] || []).filter((e) => !RP_SKIP.has(e.type));
  setHTML($("rpEvents"), evs.length
    ? evs.map((ev) => fmtEvent(ev)).join("")
    : `<div class="dim small">nothing notable at t${replay.tick}</div>`);
}

/* ------------------------------------------------------------- drawing */
function drawGrid(rows, W, H, tileOverride) {
  const water = waterFrame ? SPR.water1 : SPR.water0;
  for (let y = 0; y < H; y++) {
    const row = rows[y];
    for (let x = 0; x < W; x++) {
      const ch = (tileOverride && tileOverride.get(x + "," + y)) || row[x];
      let base = SPR.grass, over = null;
      if (ch === "b") { over = SPR.bush; }
      else if (ch === "T") { base = SPR.forestFloor; over = SPR.tree; }
      else if (ch === "t") { base = SPR.forestFloor; over = SPR.stump; }
      else if (ch === "M") { base = SPR.rock; }
      else if (ch === "m") { base = SPR.rockDepleted; }
      else if (ch === "w") { base = water; }
      ctx.drawImage(base, x * TILE, y * TILE);
      if (over) ctx.drawImage(over, x * TILE, y * TILE);
    }
  }
}
function structSprite(type) {
  return type === "fire" ? (fireFrame ? SPR.fire1 : SPR.fire0)
       : type === "farm" ? SPR.farm
       : type === "book" ? SPR.book : SPR.storehouse;
}
function drawHeat(heat, getCount) {
  let max = 1;
  for (const v of heat.values ? heat.values() : Object.values(heat)) max = Math.max(max, v);
  const entries = heat.entries ? heat.entries() : Object.entries(heat);
  for (const [k, v] of entries) {
    const [x, y] = k.split(",").map(Number);
    ctx.fillStyle = `rgba(224, 90, 78, ${0.08 + 0.5 * Math.log(1 + v) / Math.log(1 + max)})`;
    ctx.fillRect(x * TILE, y * TILE, TILE, TILE);
  }
}
function drawTrails(trails, idxOf) {
  const entries = trails.entries ? trails.entries() : Object.entries(trails);
  for (const [id, pts] of entries) {
    const col = AGENT_COLORS[idxOf(id) % AGENT_COLORS.length][0];
    for (let i = 0; i < pts.length; i++) {
      ctx.globalAlpha = 0.10 + 0.5 * (i / pts.length);
      ctx.fillStyle = col;
      ctx.fillRect(pts[i][0] * TILE + 6, pts[i][1] * TILE + 6, 4, 4);
    }
  }
  ctx.globalAlpha = 1;
}

let lastFrameMode = null;
function draw() {
  const rp = replayActive();
  if (rp !== lastFrameMode) {
    lastFrameMode = rp;
    $("mapframe").classList.toggle("replay", rp);
    $("replayBadge").classList.toggle("hidden", !rp);
  }
  ctx.fillStyle = "#06080a";
  ctx.fillRect(0, 0, map.width, map.height);

  if (rp) {
    const d = replay.data, st = replay.st;
    $("replayBadge").textContent = `⏪ REPLAY t${replay.tick} — ${d.name}`;
    drawGrid(d.grid, d.W, d.H, st.tiles);
    if (ui.heat) drawHeat(st.heat);
    if (ui.trails) drawTrails(st.trails, rIdxOf);
    for (const [k, type] of st.structures)
      ctx.drawImage(structSprite(type), +k.split(",")[0] * TILE, +k.split(",")[1] * TILE);
    for (const [id, a] of st.agents) if (!a.alive)
      ctx.drawImage(SPR.skull, a.x * TILE, a.y * TILE);
    for (const [id, a] of st.agents) if (a.alive)
      ctx.drawImage(SPR.agents[rIdxOf(id) % SPR.agents.length], a.x * TILE, a.y * TILE);
  } else if (state && state.snapshot) {
    const snap = state.snapshot;
    drawGrid(snap.grid, snap.W, snap.H, null);
    if (ui.heat && snap.heat) drawHeat(snap.heat);
    if (ui.trails && snap.trails) drawTrails(snap.trails, colorIdx);
    for (const s of snap.structures)
      ctx.drawImage(structSprite(s.type), s.x * TILE, s.y * TILE);
    // dead first so survivors draw over shared tiles
    for (const a of snap.agents) if (!a.alive)
      ctx.drawImage(SPR.skull, a.x * TILE, a.y * TILE);
    for (const a of snap.agents) if (a.alive)
      ctx.drawImage(spriteOf(a.id), a.x * TILE, a.y * TILE);

    const sel = snap.agents.find((a) => a.id === selected);
    if (sel) {
      // vision range (Chebyshev => square), clipped at map edge for clarity
      const R = snap.vis_radius;
      ctx.strokeStyle = "rgba(224,169,62,.35)";
      ctx.setLineDash([3, 3]);
      ctx.strokeRect((sel.x - R) * TILE + .5, (sel.y - R) * TILE + .5,
                     (2 * R + 1) * TILE - 1, (2 * R + 1) * TILE - 1);
      ctx.setLineDash([]);
      ctx.strokeStyle = "#e0a93e";
      ctx.strokeRect(sel.x * TILE + .5, sel.y * TILE + .5, TILE - 1, TILE - 1);
      ctx.fillStyle = "#e0a93e";
      ctx.font = "8px monospace";
      ctx.fillText(sel.id, sel.x * TILE - 1, sel.y * TILE - 3);
    }
  }
  requestAnimationFrame(draw);
}
setInterval(() => { waterFrame ^= 1; }, 900);
setInterval(() => { fireFrame ^= 1; }, 280);

/* replay color index: sorted agent ids, stable per run */
let rIdxCache = { data: null, map: {} };
function rIdxOf(id) {
  if (rIdxCache.data !== replay.data) {
    rIdxCache = { data: replay.data, map: {} };
    replay.data.agents0.map((a) => a.id).sort()
      .forEach((nm, i) => { rIdxCache.map[nm] = i; });
  }
  return rIdxCache.map[id] || 0;
}

/* ------------------------------------------------------------- map input */
function tileAt(ev) {
  const r = map.getBoundingClientRect();
  const W = replayActive() ? replay.data.W : (state && state.snapshot ? state.snapshot.W : 32);
  const x = Math.floor((ev.clientX - r.left) / r.width * W);
  const y = Math.floor((ev.clientY - r.top) / r.height * W);
  return { x, y };
}
map.addEventListener("click", (ev) => {
  if (replayActive()) return;
  if (!state || !state.snapshot) return;
  const { x, y } = tileAt(ev);
  const hit = state.snapshot.agents.find(
    (a) => Math.abs(a.x - x) <= 0 && Math.abs(a.y - y) <= 0)
    || state.snapshot.agents.find(
    (a) => Math.abs(a.x - x) <= 1 && Math.abs(a.y - y) <= 1);
  selected = hit ? hit.id : null;
  if (hit && ui.tab !== "agents" && ui.tab !== "overview") setTab("agents");
  renderPanels();
});
map.addEventListener("mousemove", (ev) => {
  const tip = $("tooltip");
  const { x, y } = tileAt(ev);
  let lines = null;
  if (replayActive()) {
    const d = replay.data, st = replay.st;
    if (x >= 0 && y >= 0 && x < d.W && y < d.H) {
      const ch = st.tiles.get(x + "," + y) || d.grid[y][x];
      lines = [`(${x}, ${y}) ${terrName(ch)}`];
      const stt = st.structures.get(x + "," + y);
      if (stt) lines.push(stt);
      for (const [id, a] of st.agents) if (a.x === x && a.y === y)
        lines.push(`${id}${a.alive ? "" : " (dead)"}`);
    }
  } else if (state && state.snapshot) {
    const snap = state.snapshot;
    if (x >= 0 && y >= 0 && x < snap.W && y < snap.H) {
      lines = [`(${x}, ${y}) ${terrName(snap.grid[y][x])}`];
      for (const s of snap.structures) if (s.x === x && s.y === y)
        lines.push(`${s.type} — built by ${s.owner}`);
      for (const a of snap.agents) if (a.x === x && a.y === y)
        lines.push(`${a.id} ${a.alive ? `⚡${a.energy}` : "(dead)"}`);
    }
  }
  if (!lines) { tip.classList.add("hidden"); return; }
  tip.textContent = lines.join("\n");
  tip.classList.remove("hidden");
  const fr = $("mapframe").getBoundingClientRect();
  tip.style.left = Math.min(ev.clientX - fr.left + 14, fr.width - 150) + "px";
  tip.style.top = (ev.clientY - fr.top + 10) + "px";
});
map.addEventListener("mouseleave", () => $("tooltip").classList.add("hidden"));
function terrName(ch) {
  return { g: "grass", b: "grass · berries", T: "forest · wood", t: "forest (cut)",
           M: "mountain · stone", m: "mountain (empty)", w: "water" }[ch] || ch;
}

/* ------------------------------------------------------------- events */
const EV_CAT = {
  said: "social", gave: "social",
  gathered: "gather", resource_depleted: "gather", spoiled: "gather",
  dropped: "gather", storehouse_deposit: "gather", storehouse_withdraw: "gather",
  ate: "gather",
  combine_succeeded: "craft", combine_failed: "craft", recipe_learned: "craft",
  tool_equipped: "craft", structure_built: "craft", structure_expired: "craft",
  farm_yielded: "craft",
  agent_died: "trouble", energy_critical: "trouble",
  planner_parse_error: "trouble", plan_interrupted: "trouble",
  planner_called: "planner", plan_completed: "planner",
};

function fmtEvent(ev) {
  const who = ev.actor || "";
  const p = ev.payload || {};
  let cls = "", text = "";
  // in the replay tab, colors come from the replayed run's roster
  const col = who ? (replayActive()
    ? AGENT_COLORS[rIdxOf(who) % AGENT_COLORS.length][0] : colorOf(who)) : "";
  const W = who ? `<span class="who" style="color:${col}">${esc(who)}</span>` : "";
  switch (ev.type) {
    case "said": cls = "speech";
      text = `${W} → ${esc(ev.target || "all")}: “${esc(p.message)}”`; break;
    case "recipe_learned": cls = "hot";
      text = `${W} learned ${esc(p.recipe)} (${esc(p.source)})`; break;
    case "combine_succeeded": cls = "hot";
      text = `${W} combine ⇒ ${esc(p.produced || p.recipe || "?")}`; break;
    case "structure_built": cls = "hot";
      text = `${W} built ${esc(p.structure || "")}`; break;
    case "tool_equipped": cls = "hot";
      text = `${W} equipped ${esc(p.tool || "")}`; break;
    case "agent_died": cls = "death"; text = `${W} died`; break;
    case "energy_critical": cls = "death"; text = `${W} energy critical`; break;
    case "planner_parse_error": cls = "death"; text = `${W} planner output unusable`; break;
    case "combine_failed": text = `${W} combine failed (${esc(p.outcome || "")})`; break;
    case "gathered": text = `${W} gathered ${p.qty} ${esc(p.resource)}`; break;
    case "ate": text = `${W} ate ${esc(p.item)} (+${p.energy})`; break;
    case "gave": text = `${W} gave ${p.qty} ${esc(p.item)} → ${esc(ev.target)}`; break;
    case "planner_called": text = `${W} replans (${esc(p.trigger)}): ${esc(p.intention || "")}`; break;
    case "plan_completed": text = `${W} finished: ${esc(p.intention || "")}`; break;
    case "plan_interrupted": text = `${W} interrupted: ${esc(p.reason || "")}`; break;
    default: {
      const kv = Object.entries(p).map(([k, v]) =>
        `${k}=${typeof v === "object" ? JSON.stringify(v) : v}`).join(" ");
      text = `${W} ${ev.type}${kv ? " " + esc(kv) : ""}`;
    }
  }
  const bord = who ? ` style="border-left-color:${col}"` : "";
  return `<div class="ev ${cls}"${bord}><span class="t">t${ev.tick}</span> ${text}</div>`;
}

/* category buttons */
for (const b of document.querySelectorAll("#evCats button"))
  b.onclick = () => {
    ui.evCat = b.dataset.cat;
    for (const x of document.querySelectorAll("#evCats button"))
      x.classList.toggle("on", x === b);
    renderPanels();
  };

const HIGHLIGHTS = new Set(["recipe_learned", "combine_succeeded", "structure_built",
  "tool_equipped", "agent_died", "said", "energy_critical"]);

/* ------------------------------------------------------------- stats canvases */
function drawSpark(canvas, arr, color) {
  const g = canvas.getContext("2d");
  g.clearRect(0, 0, canvas.width, canvas.height);
  if (!arr || !arr.length) return;
  const max = Math.max(1, ...arr);
  const bw = canvas.width / arr.length;
  g.fillStyle = color;
  for (let i = 0; i < arr.length; i++) {
    const h = arr[i] / max * (canvas.height - 6);
    g.fillRect(i * bw, canvas.height - h, Math.max(1, bw - 1), h);
  }
  g.fillStyle = "#7a8568";
  g.font = "10px monospace";
  g.fillText("max " + max, 4, 11);
}
function drawHeatmap(canvas, snap) {
  const g = canvas.getContext("2d");
  g.imageSmoothingEnabled = false;
  const W = snap.W, H = snap.H, s = canvas.width / W;
  g.fillStyle = "#0c1108";
  g.fillRect(0, 0, canvas.width, canvas.height);
  for (let y = 0; y < H; y++)
    for (let x = 0; x < W; x++) {
      const ch = snap.grid[y][x];
      g.fillStyle = ch === "w" ? "#13293a" : ch === "T" || ch === "t" ? "#16240f"
                  : ch === "M" || ch === "m" ? "#26271f" : "#15200d";
      g.fillRect(x * s, y * s, s, s);
    }
  const heat = snap.heat || {};
  let max = 1;
  for (const v of Object.values(heat)) max = Math.max(max, v);
  for (const [k, v] of Object.entries(heat)) {
    const [x, y] = k.split(",").map(Number);
    g.fillStyle = `rgba(224, 90, 78, ${0.1 + 0.75 * Math.log(1 + v) / Math.log(1 + max)})`;
    g.fillRect(x * s, y * s, s, s);
  }
}

/* ------------------------------------------------------------- panels */
function renderPanels() {
  if (!state) return;
  const snap = state.snapshot;
  const st = state.status;

  if (snap) {
    IDX = {};
    snap.agents.forEach((a, i) => { IDX[a.id] = i; });
  }

  const pill = $("statusPill");
  pill.textContent = st.toUpperCase();
  pill.className = "pill " + st;
  $("runName").textContent = state.run.name
    ? `${state.run.name} — ${state.run.provider}${state.run.model && state.run.model !== "—" ? "/" + state.run.model : ""} · ${state.run.condition} · seed ${state.run.seed}`
    : "no run loaded";

  const total = state.run.ticks || 0;
  const tick = snap ? snap.tick : 0;
  const pct = total ? Math.min(tick / total * 100, 100) : 0;
  sampleRate(state.run.name, st, tick);
  const eta = st === "running" ? etaText(total, tick) : "";
  $("progFill").style.width = pct + "%";
  $("progress").classList.toggle("done", st === "finished");
  let ptxt = "no run";
  if (snap) {
    ptxt = `TICK ${tick} / ${total}  ·  ${pct.toFixed(pct > 0 && pct < 10 ? 1 : 0)}%`;
    if (eta) ptxt += `  ·  ${eta.rate.toFixed(1)}s/tick  ·  ETA ~${eta.eta}`;
    if (st === "finished") ptxt += "  ·  DONE";
    if (st === "paused") ptxt += "  ·  PAUSED";
    const started = state.run.started;
    if (started && st === "running") ptxt += `  ·  elapsed ${fmtDur(Date.now() / 1000 - started)}`;
  }
  $("progText").textContent = ptxt;

  $("btnPause").disabled = st !== "running";
  $("btnResume").disabled = st !== "paused";
  $("btnStop").disabled = st !== "running" && st !== "paused";

  const ext = state.external || [];
  const eb = $("externalBanner");
  if (ext.length) {
    eb.textContent = `⚠ external run active in runs/${ext.join(", runs/")} — starting an Ollama run now will share the GPU and slow both`;
    eb.classList.remove("hidden");
  } else eb.classList.add("hidden");

  if (state.error) { $("errorBanner").textContent = state.error; $("errorBanner").classList.remove("hidden"); }

  if (!snap) { setHTML($("keyStats"), ""); return; }
  const s = snap.stats;
  const nDisc = Object.keys(s.discovered).length;
  setHTML($("keyStats"), `
    <span class="ks ${s.alive < snap.agents.length ? "warm" : ""}">ALIVE<b>${s.alive}/${snap.agents.length}</b></span>
    <span class="ks ${s.deaths ? "bad" : ""}">DEATHS<b>${s.deaths}</b></span>
    <span class="ks ${s.parse_rate >= 5 ? "bad" : ""}">PARSE FAIL<b>${s.parse_rate}%</b></span>
    <span class="ks ${nDisc ? "warm" : ""}">DISCOVERED<b>${nDisc}/${s.recipes.length}</b></span>
    <span class="ks">LLM CALLS<b>${s.planner_calls}</b></span>
    ${s.spend ? `<span class="ks ${s.spend.rate_limited ? "warm" : ""}">SPEND<b>$${s.spend.cost_usd.toFixed(4)}</b></span>` : ""}`);

  // only the visible pane needs work
  if (ui.tab === "overview") renderOverview(snap, s);
  else if (ui.tab === "agents") renderAgents(snap);
  else if (ui.tab === "events") renderEvents(snap);
  else if (ui.tab === "stats") renderStats(snap, s);
}

/* ---------------- overview */
function renderOverview(snap, s) {
  const changed = setHTML($("ovAgents"), snap.agents.map((a, i) => {
    const cls = (a.id === selected ? " sel" : "") + (a.alive ? "" : " dead");
    const elvl = a.energy < 20 ? "crit" : a.energy < 40 ? "low" : "";
    const inv = Object.entries(a.inventory).sort().map(([k, v]) =>
      `<canvas data-item="${esc(k)}" title="${esc(k)} ×${v}"></canvas>`).join("");
    return `<div class="mini-card${cls}" data-id="${esc(a.id)}">
      <canvas class="portrait" data-i="${i}" style="width:34px;height:34px"></canvas>
      <div class="pcol">
        <span class="nm" style="color:${colorOf(a.id)}">${esc(a.id)}</span>
        <div class="ebar ${elvl}"><i style="width:${Math.max(a.energy, 0)}%"></i></div>
        <div class="inv">${a.alive ? inv : "☠"}</div>
      </div>
    </div>`;
  }).join(""));
  if (changed) {
    blitArt($("ovAgents"));
    for (const el of $("ovAgents").querySelectorAll(".mini-card"))
      el.onclick = () => { selected = el.dataset.id; setTab("agents"); };
  }

  setHTML($("ovDiscoveries"), s.recipes.map((r) => {
    const d = s.discovered[r];
    return `<span class="chip ${d ? "lit" : ""}" ${d ? `title="${d.source} by ${d.by} at t${d.tick}"` : ""}>` +
           `${r}${d ? ` ${esc(d.by)} t${d.tick}${d.source === "taught" ? " ✎" : ""}` : ""}</span>`;
  }).join(""));

  const lv = s.live;
  setHTML($("ovStats"), [
    ["alive", `${s.alive}/${snap.agents.length}`, s.alive < snap.agents.length ? "warm" : ""],
    ["deaths", s.deaths, s.deaths ? "bad" : ""],
    ["LLM calls", s.planner_calls, ""],
    ["parse fail", s.parse_rate + "%", s.parse_rate >= 5 ? "bad" : ""],
    ["plan completion", lv.completion_pct + "%", ""],
    ["gathers /100t", lv.gathers_per_100, ""],
    ["taught", s.taught, s.taught ? "warm" : ""],
    ["said / gave", `${s.said} / ${s.gave}`, ""],
  ].map(([k, v, c]) =>
    `<div class="stat ${c}"><span>${k}</span><b>${v}</b></div>`).join(""));

  setHTML($("ovEvents"), snap.events.filter((e) => HIGHLIGHTS.has(e.type))
    .slice(-12).reverse().map(fmtEvent).join("") ||
    `<div class="dim small">nothing notable yet</div>`);
}

/* ---------------- agents */
function renderAgents(snap) {
  const changed = setHTML($("agentList"), snap.agents.map((a, i) => {
    const cls = (a.id === selected ? " sel" : "") + (a.alive ? "" : " dead");
    const elvl = a.energy < 20 ? "crit" : a.energy < 40 ? "low" : "";
    const beliefs = a.beliefs.length
      ? `<ul class="beliefs">${a.beliefs.map((b) => `<li>${esc(b)}</li>`).join("")}</ul>`
      : `<div class="dim small">(no beliefs yet)</div>`;
    const plan = a.plan.length
      ? `<ul class="plan">${a.plan.map((stp, j) =>
          `<li class="${j === 0 ? "now" : ""}">${esc(fmtStep(stp))}</li>`).join("")}</ul>`
      : `<div class="dim small">(no queued steps — will replan next tick)</div>`;
    const pr = (state.prompts || {})[a.id];
    let llm = `<div class="dim small">(no planner call recorded yet)</div>`;
    if (pr) {
      llm = `<div class="small">t${pr.tick} · attempt ${pr.attempt} · ${pr.ms} ms${pr.model ? ` · ${esc(pr.model)}` : ""} ·
        ${pr.ok ? `<span class="ok">parsed ok</span>` : `<span class="fail">${esc(pr.error)}</span>`}</div>
        <details data-k="prompt-${i}"><summary>prompt (what the model saw)</summary><pre>${esc(pr.system)}\n\n${esc(pr.prompt)}</pre></details>
        <details data-k="resp-${i}"><summary>raw response</summary><pre>${esc(pr.response || "(no response)")}</pre></details>`;
    }
    return `<div class="acard${cls}" data-id="${esc(a.id)}">
      <div class="head">
        <canvas class="portrait" data-i="${i}" style="width:44px;height:44px"></canvas>
        <div class="meta">
          <span class="nm"><span class="dot" style="background:${colorOf(a.id)}"></span>
            ${esc(a.id)} <span class="pos">(${a.x}, ${a.y}) · planned t${a.last_plan_tick}</span></span>
          <div class="ebar ${elvl}"><i style="width:${Math.max(a.energy, 0)}%"></i></div>
          <span class="pos">${a.alive ? `⚡ ${a.energy}/100` : "☠ dead"}</span>
        </div>
      </div>
      ${slotsHTML(a.inventory, a.tools, capacityOf(a))}
      <div class="what">${a.alive ? esc(a.intention || "…") : ""}</div>
      ${beliefs}
      <details data-k="plan-${i}"><summary>plan queue (${a.plan.length})</summary>${plan}</details>
      <details data-k="know-${i}"><summary>knows ${a.knowledge.length ? `(${a.knowledge.length})` : "(none)"}</summary>
        <div class="small">${a.knowledge.join(", ") || `<span class="dim">no confirmed recipes</span>`}
        ${a.heard.length ? `<br>heard of: ${a.heard.join(", ")}` : ""}</div></details>
      <details data-k="notes-${i}"><summary>notes (scratchpad)</summary><pre>${esc(a.notes || "(empty)")}</pre></details>
      <details data-k="journal-${i}"><summary>journal — what ${esc(a.id)} has seen</summary>
        <pre>${esc(a.journal.join("\n") || "(empty)")}</pre></details>
      <details data-k="llm-${i}"><summary>last llm call</summary>${llm}</details>
    </div>`;
  }).join(""));
  if (changed) {
    blitArt($("agentList"));
    for (const el of $("agentList").querySelectorAll(".acard .head"))
      el.onclick = () => { selected = el.parentElement.dataset.id; renderPanels(); };
  }
}

/* ---------------- events */
function renderEvents(snap) {
  const chipsChanged = setHTML($("evFilterAgents"),
    snap.agents.map((a) => {
      const on = !ui.evAgents || ui.evAgents.has(a.id);
      return `<span class="chip click ${on ? "" : "off"}" data-id="${esc(a.id)}"
        style="border-color:${colorOf(a.id)};color:${on ? colorOf(a.id) : ""}">${esc(a.id)}</span>`;
    }).join("") +
    `<span class="chip click" data-id="*">all</span>`);
  if (chipsChanged)
    for (const c of $("evFilterAgents").querySelectorAll(".chip"))
      c.onclick = () => {
        const id = c.dataset.id;
        if (id === "*") ui.evAgents = null;
        else {
          if (!ui.evAgents) ui.evAgents = new Set(snap.agents.map((a) => a.id));
          if (ui.evAgents.has(id)) ui.evAgents.delete(id); else ui.evAgents.add(id);
          if (ui.evAgents.size === snap.agents.length) ui.evAgents = null;
        }
        $("evFilterAgents")._lastHTML = null;   // force chip refresh
        renderPanels();
      };

  const evs = snap.events.filter((e) => {
    if (ui.evCat !== "all" && EV_CAT[e.type] !== ui.evCat) return false;
    if (ui.evAgents && e.actor && !ui.evAgents.has(e.actor)) return false;
    if (ui.evAgents && !e.actor) return false;
    return true;
  });
  setHTML($("evList"), evs.slice().reverse().map(fmtEvent).join("") ||
    `<div class="dim small">no matching events (the feed keeps the last 250)</div>`);
}

/* ---------------- stats */
function renderStats(snap, s) {
  const lv = s.live, bl = state.baseline;
  const lat = s.latency_ms;
  const gb = s.gathered_by || {};
  const gmax = Math.max(1, ...Object.values(gb));
  const html = `
    <h2>▸ This run</h2>
    <div class="statgrid">
      ${[["alive", `${s.alive}/${snap.agents.length}`, ""],
         ["deaths", s.deaths, s.deaths ? "bad" : ""],
         ["LLM calls", s.planner_calls, ""],
         ["calls / tick", s.calls_per_tick, ""],
         ["parse fail", s.parse_rate + "%", s.parse_rate >= 5 ? "bad" : ""],
         ["plan completion", lv.completion_pct + "%", ""],
         ["taught recipes", s.taught, s.taught ? "warm" : ""],
         ["discovered", `${lv.recipes_discovered}/${s.recipes.length}`, lv.recipes_discovered ? "warm" : ""],
         ["combines ok / fail", `${s.combine_succeeded} / ${s.combine_failed}`, ""],
         ["said / gave", `${s.said} / ${s.gave}`, ""],
        ].map(([k, v, c]) => `<div class="stat ${c}"><span>${k}</span><b>${v}</b></div>`).join("")}
    </div>
    <h2>▸ Planning time</h2>
    <div class="statgrid">
      ${lat ? [["calls measured", lat.n], ["avg", lat.avg + " ms"],
               ["p50", lat.p50 + " ms"], ["p95", lat.p95 + " ms"]]
          .map(([k, v]) => `<div class="stat"><span>${k}</span><b>${v}</b></div>`).join("")
        : `<div class="dim small">no LLM latency yet (mock runs have none)</div>`}
    </div>
    ${s.spend ? `<h2>▸ Spend</h2><div class="statgrid">
      <div class="stat"><span>cost</span><b>$${s.spend.cost_usd.toFixed(4)}</b></div>
      <div class="stat"><span>tokens in/out</span><b>${s.spend.prompt_tokens}/${s.spend.completion_tokens}</b></div>
      <div class="stat ${s.spend.rate_limited ? "warm" : ""}"><span>rate-limited</span><b>${s.spend.rate_limited}</b></div>
    </div>` : ""}
    <h2>▸ Gathered, by resource</h2>
    <div class="bars">${Object.entries(gb).sort().map(([k, v]) =>
      `<div class="bar"><span>${esc(k)}</span><i style="width:${v / gmax * 60}%"></i><b>${v}</b></div>`).join("") ||
      `<div class="dim small">nothing gathered yet</div>`}
    </div>
    <h2>▸ vs no-LLM runs</h2>
    ${!bl ? `<div class="basenote">no completed no-LLM runs found in runs/</div>`
      : `<table class="basetable">
          <tr><th></th><th>this run</th><th>no-LLM (${bl.runs.length} runs, ${bl.ticks}t)</th></tr>
          <tr><td>deaths /1k ticks</td><td>${lv.deaths_per_1k}</td><td>${bl.deaths_per_1k}</td></tr>
          <tr><td>plan completion</td><td>${lv.completion_pct}%</td><td>${bl.completion_pct}%</td></tr>
          <tr><td>gathers /100 ticks</td><td>${lv.gathers_per_100}</td><td>${bl.gathers_per_100}</td></tr>
          <tr><td>recipes found</td><td>${lv.recipes_discovered}</td><td>${bl.recipes_discovered}</td></tr>
        </table>`}`;
  setHTML($("statsBody"), html);
  if (snap.series) {
    drawSpark($("sparkCalls"), snap.series.calls, "#9fd45e");
    drawSpark($("sparkGathers"), snap.series.gathers, "#e0a93e");
  }
  drawHeatmap($("heatmap"), snap);
}

function fmtStep(s) {
  switch (s.action) {
    case "move_to": return `move_to (${s.x}, ${s.y})`;
    case "gather": return `gather ${s.resource}` + (s.x != null ? ` @ (${s.x}, ${s.y})` : "");
    case "eat": return `eat ${s.item}`;
    case "combine": return `combine ${Object.entries(s.items).map(([k, v]) => `${v} ${k}`).join(" + ")}`;
    case "say": return `say → ${s.target || "all"}: “${s.message}”`;
    case "give": return `give ${s.qty} ${s.item} → ${s.target}`;
    default: return Object.values(s).join(" ");
  }
}

/* ------------------------------------------------------------- run setup */
let WORLD_DEFAULTS = {}, DEFAULT_NAMES = [];
fetch("/api/defaults").then((r) => r.json()).then((d) => {
  WORLD_DEFAULTS = d.world;
  DEFAULT_NAMES = d.default_names || [];
  $("worldForm").innerHTML = Object.entries(d.world).map(([k, v]) =>
    `<label class="half">${k.replaceAll("_", " ")}
       <input class="wparam" data-key="${k}" type="number"
              value="${v}" step="${Number.isInteger(v) ? 1 : 0.01}"></label>`).join("");
  buildNamesGrid();
});

function currentNames() {
  return Array.from(document.querySelectorAll("#namesGrid input")).map((i) => i.value.trim());
}
function buildNamesGrid(shuffled) {
  const n = Math.max(1, Math.min(12, +$("fAgents").value || 10));
  const prev = currentNames();
  let pool = DEFAULT_NAMES.slice();
  if (shuffled)
    for (let i = pool.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [pool[i], pool[j]] = [pool[j], pool[i]];
    }
  $("namesGrid").innerHTML = Array.from({ length: n }, (_, i) => {
    const v = shuffled ? (pool[i] || `A${String(i).padStart(2, "0")}`)
            : (prev[i] || pool[i] || `A${String(i).padStart(2, "0")}`);
    return `<label><b style="color:${AGENT_COLORS[i % AGENT_COLORS.length][0]}">#${i}</b>
      <input maxlength="14" value="${esc(v)}"></label>`;
  }).join("");
}
$("fAgents").onchange = () => buildNamesGrid();
$("btnShuffleNames").onclick = () => buildNamesGrid(true);

function namesFor(n) {
  const have = currentNames().filter(Boolean);
  const out = [];
  for (let i = 0; i < n; i++)
    out.push(have[i] || DEFAULT_NAMES[i] || `A${String(i).padStart(2, "0")}`);
  return out;
}

$("btnMock").onclick = () =>
  post("/api/start", { preset: "mock", condition: "blind", agents: 10,
                       ticks: 300, seed: 42, tick_delay: 0.12,
                       agent_names: namesFor(10) });
$("btnAcceptance").onclick = () =>
  post("/api/start", { preset: "local-e2b", condition: "blind",
                       agents: 10, ticks: 1000, seed: 42,
                       agent_names: namesFor(10),
                       name: "m3_accept_ui_" + new Date().toISOString().slice(11, 19).replaceAll(":", "") });

$("fPreset").onchange = () => {
  const custom = $("fPreset").value === "custom";
  $("rowProvider").classList.toggle("hidden", !custom);
  $("rowModel").classList.toggle("hidden", !custom);
};
$("btnCustom").onclick = () => {
  const world = {};
  for (const inp of document.querySelectorAll("#worldForm .wparam")) {
    if (+inp.value !== WORLD_DEFAULTS[inp.dataset.key]) world[inp.dataset.key] = +inp.value;
  }
  const n = +$("fAgents").value;
  post("/api/start", {
    preset: $("fPreset").value,
    provider: $("fProvider").value, model: $("fModel").value || null,
    condition: $("fCondition").value, agents: n,
    ticks: +$("fTicks").value, seed: +$("fSeed").value,
    name: $("fName").value || null,
    agent_names: namesFor(n),
    world,
  });
};
$("btnPause").onclick = () => post("/api/pause");
$("btnResume").onclick = () => post("/api/resume");
$("btnStop").onclick = () => post("/api/stop");
$("speed").oninput = (e) => {
  $("speedVal").textContent = e.target.value + " ms";
  post("/api/speed", { tick_delay: e.target.value / 1000 });
};

/* ------------------------------------------------------------- files */
let runsCache = null;
function loadRuns() {
  fetch("/api/runs").then((r) => r.json()).then((runs) => {
    runsCache = runs;
    renderRuns();
  }).catch((e) => showError("runs list failed: " + e));
}
$("filesRefresh").onclick = loadRuns;
let runOpen = null;

function renderRuns() {
  if (!runsCache) return;
  setHTML($("runsList"), runsCache.map((r) => {
    const sm = r.summary, sc = r.scan;
    const outcome = sm ? sm.outcome : "no summary";
    const facts = sm
      ? `<b>${sm.ticks_done}</b> ticks · <b>${sm.agents.length}</b> agents · ` +
        `${sm.counters.deaths || 0} deaths · ${Object.keys(sm.discovered).length} discovered · ` +
        `${esc(sm.provider || "?")} · ${fmtDur(sm.duration_s)}`
      : sc
      ? `<b>${sc.ticks_done}</b> ticks · ${sc.agents_seen} agents · ${sc.deaths} deaths · ` +
        `${Object.keys(sc.discovered).length} discovered`
      : "";
    let detail = "";
    if (runOpen === r.name) {
      const disc = sm ? sm.discovered : (sc ? sc.discovered : {});
      const discHtml = Object.keys(disc).length
        ? Object.entries(disc).map(([k, v]) =>
            `<span class="chip lit">${esc(k)} t${typeof v === "object" ? v.tick : v}</span>`).join(" ")
        : `<span class="dim">none</span>`;
      const agentsHtml = sm ? `<table class="basetable">
          <tr><th>agent</th><th>fate</th><th>energy</th><th>knows</th></tr>
          ${sm.agents.map((a) => `<tr><td>${esc(a.id)}</td>
            <td>${a.alive ? "alive" : "died"}</td><td>${a.energy}</td>
            <td>${a.knowledge.join(", ") || "—"}</td></tr>`).join("")}
        </table>` : "";
      const latHtml = sm && sm.latency_ms
        ? `<div class="dim small">planner: ${sm.latency_ms.n} calls · avg ${sm.latency_ms.avg} ms · p95 ${sm.latency_ms.p95} ms</div>` : "";
      const spendHtml = sm && sm.spend
        ? `<div class="dim small">spend: $${sm.spend.cost_usd.toFixed(4)} · rate-limited ${sm.spend.rate_limited}</div>` : "";
      detail = `<div class="detail">
        <div class="chips">${discHtml}</div>
        ${latHtml}${spendHtml}${agentsHtml}
        <div class="actions"><button class="mini rp" data-run="${esc(r.name)}">⏪ OPEN IN REPLAY</button></div>
      </div>`;
    }
    return `<div class="runrow" data-run="${esc(r.name)}">
      <div class="top"><span class="rn">${esc(r.name)}</span>
        <span class="outcome ${esc(outcome)}">${esc(outcome)}</span>
        <span class="date">${fmtDate(r.mtime)}</span></div>
      <div class="facts">${facts}</div>
      ${detail}
    </div>`;
  }).join("") || `<div class="dim small">no runs found in runs/</div>`);
  for (const row of $("runsList").querySelectorAll(".runrow"))
    row.onclick = (e) => {
      if (e.target.classList.contains("rp")) return;
      runOpen = runOpen === row.dataset.run ? null : row.dataset.run;
      $("runsList")._lastHTML = null;
      renderRuns();
    };
  for (const b of $("runsList").querySelectorAll("button.rp"))
    b.onclick = () => {
      setTab("replay");
      rpFillRunSelect();
      $("rpRun").value = b.dataset.run;
      rpLoad(b.dataset.run);
    };
}

/* ------------------------------------------------------------- about */
$("aboutBody").innerHTML = `
<h2>▸ What is this?</h2>
<p><b>Terra Incognita</b> is a multi-agent survival experiment: a society of LLM
agents dropped onto a 32×32 island whose technology rules are <b>hidden</b>.
The question is not whether they survive — the scripted no-LLM baseline does
that — but whether they <b>discover</b> the tech tree by experimenting, and
whether knowledge <b>spreads socially</b>: one agent figures out the axe,
tells a neighbour, and the neighbour builds one without rediscovering it
(that's a <i>taught</i> recipe, the headline metric).</p>

<h2>▸ What the agents are told at start</h2>
<blockquote>You are {name}, a survivor in an unknown world. … your energy drains
every tick and at 0 energy you die, permanently. Berries are food — eat(berries)
restores 25 energy. Wood and stone are materials, not food. … Beyond that you do
NOT know how this world works. You can discover how to create things by
combining items, but failures may destroy ingredients. You can talk to nearby
agents. Your goals: survive, and understand this world.</blockquote>
<p>Everything else — recipes, tile conditions, tool effects — they must find
out. In the <code>blind</code> condition nobody knows anything; <code>oracle</code>
gives everyone the full tree; <code>fragmented</code> gives each agent one
random recipe; <code>one_expert</code> gives one agent everything.</p>

<h2>▸ How their context works</h2>
<p>Each planner call sends one prompt assembled from engine state — the agent
has no other memory between calls:</p>
<ul>
<li><b>STATUS</b> — energy, position, inventory slots, tools</li>
<li><b>YOU KNOW</b> — confirmed recipes (crafted at least once)</li>
<li><b>YOU BELIEVE</b> — hypotheses the model wrote for itself last time</li>
<li><b>NOTES</b> — a private scratchpad (max 1500 chars) it rewrites each call</li>
<li><b>NEARBY</b> — visible resources, structures, agents (vision radius 6)</li>
<li><b>RECENT EVENTS</b> — its journal: what it saw and what happened to it</li>
<li><b>ACTIONS</b> — the tool list below</li>
</ul>
<p>Beliefs and notes are <i>mind-owned</i>: the engine stores them verbatim and
never reads them. They are the agent's only long-term memory.</p>

<h2>▸ Tools (actions)</h2>
<table>
<tr><th>action</th><th>what it does</th></tr>
<tr><td><code>move_to(x, y)</code></td><td>walk toward a tile (8-direction, pathfinding around water)</td></tr>
<tr><td><code>gather(resource, x?, y?)</code></td><td>walk to the tile and collect berries / wood / stone</td></tr>
<tr><td><code>eat(item)</code></td><td>berries +25 energy, cooked food +50</td></tr>
<tr><td><code>combine({item: qty})</code></td><td>the discovery mechanism — try a recipe; failures can destroy materials</td></tr>
<tr><td><code>say(target, message)</code></td><td>talk to agents within 3 tiles; being addressed forces them to replan</td></tr>
<tr><td><code>give(target, item, qty)</code></td><td>hand items to an adjacent agent</td></tr>
<tr><td><code>drop / deposit / withdraw</code></td><td>inventory and storehouse management</td></tr>
<tr><td><code>wait</code></td><td>do nothing this tick</td></tr>
</table>

<h2>▸ Why plans?</h2>
<p>Calling an LLM every tick for every agent would be slow and expensive, and
would reward twitch reactions over thinking. Instead the model plans
<b>up to 20 steps</b>, and a cheap executor (Tier B) carries them out tick by
tick. The planner is consulted again only when the plan finishes, a step fails,
someone speaks to the agent, or a 15-tick timer fires. A survival reflex eats
food automatically below 30 energy. This split — slow deliberate planner, fast
dumb executor — is the architecture being tested.</p>

<h2>▸ The hidden tech tree (spoilers)</h2>
<p>axe · pickaxe · basket (tier 1) → fire · farm (tier 2) → cooked food ·
storehouse (tier 3). Conditions make it non-obvious: fire needs the same
ingredients as a pickaxe but standing on grass; cooked food needs a fire
adjacent; the farm needs an axe in hand.</p>

<h2>▸ What's next (imminent goals)</h2>
<ul>
<li><b>M3 acceptance run</b> — 10 agents × 1000 ticks on a local model:
no crash, &lt;5% parse errors, ≥1 taught recipe</li>
<li><b>OpenRouter model mixes</b> — e.g. 90% free gemma-4 / 10% paid qwen3.6,
does a smarter minority lift the society?</li>
<li><b>Multi-game statistics</b> — aggregate metrics across many seeded runs,
not just one story</li>
<li><b>Reproduction &amp; combat</b> — population dynamics and conflict</li>
<li><b>More structures &amp; technologies</b> — deeper tree, longer horizons</li>
</ul>`;

/* ------------------------------------------------------------- legend */
(function legend() {
  const items = [["grass", SPR.grass], ["berries", SPR.bush], ["forest", SPR.tree],
    ["stone", SPR.rock], ["water", SPR.water0], ["fire", SPR.fire0],
    ["farm", SPR.farm], ["storehouse", SPR.storehouse], ["book", SPR.book],
    ["agent", SPR.agents[0]], ["died here", SPR.skull]];
  $("legend").innerHTML = items.map(([n]) => `<span><canvas data-n="${n}" width="16" height="16"></canvas>${n}</span>`).join("");
  const by = Object.fromEntries(items);
  for (const c of document.querySelectorAll("#legend canvas")) {
    const g = c.getContext("2d");
    if (by[c.dataset.n] !== SPR.bush && by[c.dataset.n] !== SPR.tree)
      g.drawImage(SPR.grass, 0, 0);
    if (c.dataset.n === "berries" || c.dataset.n === "forest") g.drawImage(SPR.forestFloor, 0, 0);
    g.drawImage(by[c.dataset.n], 0, 0);
  }
})();

setTab(ui.tab);
poll();
draw();
