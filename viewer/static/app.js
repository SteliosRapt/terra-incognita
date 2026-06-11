/* app.js — polls /api/state, draws the world, runs the panels. */
"use strict";

const SPR = buildSprites();
const TILE = 16;
const POLL_MS = 700;

let state = null;        // latest /api/state payload
let selected = null;     // selected agent id
let waterFrame = 0, fireFrame = 0;

const $ = (id) => document.getElementById(id);
const map = $("map"), ctx = map.getContext("2d");
ctx.imageSmoothingEnabled = false;

/* ------------------------------------------------------------- helpers */
function colorOf(id) {
  const i = parseInt(id.slice(1), 10) || 0;
  return AGENT_COLORS[i % AGENT_COLORS.length][0];
}
function spriteOf(id) {
  const i = parseInt(id.slice(1), 10) || 0;
  return SPR.agents[i % SPR.agents.length];
}
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

/* ------------------------------------------------------------- polling */
async function poll() {
  try {
    const r = await fetch("/api/state");
    state = await r.json();
    renderPanels();
  } catch (e) { /* server briefly busy — keep last state */ }
  setTimeout(poll, POLL_MS);
}

/* ------------------------------------------------------------- drawing */
function draw() {
  const snap = state && state.snapshot;
  ctx.fillStyle = "#06080a";
  ctx.fillRect(0, 0, map.width, map.height);
  if (snap) {
    const water = waterFrame ? SPR.water1 : SPR.water0;
    for (let y = 0; y < snap.H; y++) {
      const row = snap.grid[y];
      for (let x = 0; x < snap.W; x++) {
        const ch = row[x];
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
    for (const s of snap.structures) {
      const spr = s.type === "fire" ? (fireFrame ? SPR.fire1 : SPR.fire0)
                : s.type === "farm" ? SPR.farm : SPR.storehouse;
      ctx.drawImage(spr, s.x * TILE, s.y * TILE);
    }
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

/* ------------------------------------------------------------- map input */
function tileAt(ev) {
  const r = map.getBoundingClientRect();
  const x = Math.floor((ev.clientX - r.left) / r.width * 32);
  const y = Math.floor((ev.clientY - r.top) / r.height * 32);
  return { x, y };
}
map.addEventListener("click", (ev) => {
  if (!state || !state.snapshot) return;
  const { x, y } = tileAt(ev);
  const hit = state.snapshot.agents.find(
    (a) => Math.abs(a.x - x) <= 0 && Math.abs(a.y - y) <= 0)
    || state.snapshot.agents.find(
    (a) => Math.abs(a.x - x) <= 1 && Math.abs(a.y - y) <= 1);
  selected = hit ? hit.id : null;
  renderPanels();
});
map.addEventListener("mousemove", (ev) => {
  const tip = $("tooltip");
  if (!state || !state.snapshot) { tip.classList.add("hidden"); return; }
  const snap = state.snapshot;
  const { x, y } = tileAt(ev);
  if (x < 0 || y < 0 || x >= snap.W || y >= snap.H) { tip.classList.add("hidden"); return; }
  const ch = snap.grid[y][x];
  const terr = { g: "grass", b: "grass · berries", T: "forest · wood", t: "forest (cut)",
                 M: "mountain · stone", m: "mountain (empty)", w: "water" }[ch] || ch;
  const lines = [`(${x}, ${y}) ${terr}`];
  for (const s of snap.structures) if (s.x === x && s.y === y)
    lines.push(`${s.type} — built by ${s.owner}`);
  for (const a of snap.agents) if (a.x === x && a.y === y)
    lines.push(`${a.id} ${a.alive ? `⚡${a.energy}` : "(dead)"}`);
  tip.textContent = lines.join("\n");
  tip.classList.remove("hidden");
  const fr = $("mapframe").getBoundingClientRect();
  tip.style.left = Math.min(ev.clientX - fr.left + 14, fr.width - 150) + "px";
  tip.style.top = (ev.clientY - fr.top + 10) + "px";
});
map.addEventListener("mouseleave", () => $("tooltip").classList.add("hidden"));

/* ------------------------------------------------------------- panels */
function renderPanels() {
  if (!state) return;
  const snap = state.snapshot;
  const st = state.status;

  const pill = $("statusPill");
  pill.textContent = st.toUpperCase();
  pill.className = "pill " + st;
  $("runName").textContent = state.run.name
    ? `${state.run.name} — ${state.run.provider}${state.run.model && state.run.model !== "—" ? "/" + state.run.model : ""} · ${state.run.condition} · seed ${state.run.seed}`
    : "no run loaded";

  const total = state.run.ticks || 0;
  const tick = snap ? snap.tick : 0;
  const pct = total ? Math.min(tick / total * 100, 100) : 0;
  $("progFill").style.width = pct + "%";
  $("progress").classList.toggle("done", st === "finished");
  $("progText").textContent = snap
    ? `TICK ${tick} / ${total}  ·  ${pct.toFixed(0)}%` + (st === "finished" ? "  ·  DONE" : st === "paused" ? "  ·  PAUSED" : "")
    : "no run";

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

  if (!snap) { $("keyStats").innerHTML = ""; return; }
  const s = snap.stats;
  const nDisc = Object.keys(s.discovered).length;
  $("keyStats").innerHTML = `
    <span class="ks ${s.alive < snap.agents.length ? "warm" : ""}">ALIVE<b>${s.alive}/${snap.agents.length}</b></span>
    <span class="ks ${s.deaths ? "bad" : ""}">DEATHS<b>${s.deaths}</b></span>
    <span class="ks ${s.parse_rate >= 5 ? "bad" : ""}">PARSE FAIL<b>${s.parse_rate}%</b></span>
    <span class="ks ${s.taught ? "warm" : ""}">TAUGHT<b>${s.taught}</b></span>
    <span class="ks ${nDisc ? "warm" : ""}">DISCOVERED<b>${nDisc}/${s.recipes.length}</b></span>
    <span class="ks">LLM CALLS<b>${s.planner_calls}</b></span>`;

  $("discoveries").innerHTML = s.recipes.map((r) => {
    const t = s.discovered[r];
    return `<span class="chip ${t !== undefined ? "lit" : ""}">${r}${t !== undefined ? " t" + t : ""}</span>`;
  }).join("");

  $("feed").innerHTML = snap.events.slice().reverse().map(fmtEvent).join("");

  $("agentCards").innerHTML = snap.agents.map((a) => {
    const cls = (a.id === selected ? " sel" : "") + (a.alive ? "" : " dead");
    const elvl = a.energy < 20 ? "crit" : a.energy < 40 ? "low" : "";
    return `<div class="card${cls}" data-id="${a.id}">
      <div class="row"><span class="aid">${a.id}</span>
        <span class="dot" style="background:${colorOf(a.id)}"></span></div>
      <div class="ebar ${elvl}"><i style="width:${Math.max(a.energy, 0)}%"></i></div>
      <div class="what">${a.alive ? esc(a.intention || "…") : "☠ dead"}</div>
    </div>`;
  }).join("");
  for (const el of document.querySelectorAll(".card"))
    el.onclick = () => { selected = el.dataset.id; renderPanels(); };

  renderInspector(snap);
}

function fmtEvent(ev) {
  const who = ev.actor || "";
  const p = ev.payload || {};
  let cls = "", text = "";
  switch (ev.type) {
    case "said": cls = "speech";
      text = `${who} → ${ev.target || "all"}: “${esc(p.message)}”`; break;
    case "recipe_learned": cls = "hot";
      text = `${who} learned ${p.recipe} (${p.source})`; break;
    case "combine_succeeded": cls = "hot";
      text = `${who} combine ⇒ ${p.result || p.recipe || "?"}`; break;
    case "structure_built": cls = "hot";
      text = `${who} built ${p.structure || p.result || ""}`; break;
    case "tool_equipped": cls = "hot";
      text = `${who} equipped ${p.tool || p.result || ""}`; break;
    case "agent_died": cls = "death"; text = `${who} died`; break;
    case "energy_critical": cls = "death"; text = `${who} energy critical`; break;
    case "planner_parse_error": cls = "death"; text = `${who} planner output unusable`; break;
    case "combine_failed": text = `${who} combine failed`; break;
    case "planner_called": text = `${who} replans (${p.trigger}): ${esc(p.intention || "")}`; break;
    default: {
      const kv = Object.entries(p).map(([k, v]) =>
        `${k}=${typeof v === "object" ? JSON.stringify(v) : v}`).join(" ");
      text = `${who} ${ev.type}${kv ? " " + esc(kv) : ""}`;
    }
  }
  return `<div class="ev ${cls}"><span class="t">t${ev.tick}</span> ${text}</div>`;
}

function renderInspector(snap) {
  const box = $("inspector");
  const a = snap.agents.find((x) => x.id === selected);
  $("inspId").textContent = a ? a.id : "";
  if (!a) { box.innerHTML = `<p class="dim">select an agent on the map or in the list</p>`; return; }

  const inv = Object.entries(a.inventory).map(([k, v]) => `${k} ×${v}`).join(", ") || "empty";
  const plan = a.plan.length
    ? `<ul>${a.plan.map((st, i) =>
        `<li class="${i === 0 ? "now" : ""}">${esc(fmtStep(st))}</li>`).join("")}</ul>`
    : `<p class="dim">(no queued steps — will replan next tick)</p>`;
  const beliefs = a.beliefs.length
    ? `<ul>${a.beliefs.map((b) => `<li>${esc(b)}</li>`).join("")}</ul>`
    : `<p class="dim">(none)</p>`;

  const pr = (state.prompts || {})[a.id];
  let llm = `<p class="dim">(no planner call recorded yet)</p>`;
  if (pr) {
    llm = `<p class="kv">t${pr.tick} · attempt ${pr.attempt} · ${pr.ms} ms ·
      ${pr.ok ? `<span class="ok">parsed ok</span>` : `<span class="fail">${esc(pr.error)}</span>`}</p>
      <details><summary>prompt (what the model saw)</summary><pre>${esc(pr.system)}\n\n${esc(pr.prompt)}</pre></details>
      <details><summary>raw response</summary><pre>${esc(pr.response || "(no response)")}</pre></details>`;
  }

  box.innerHTML = `
    <p class="kv">${a.alive ? `⚡ <b>${a.energy}</b>/100` : "☠ dead"} · pos <b>(${a.x}, ${a.y})</b>
      · last planned t${a.last_plan_tick}</p>
    <h4>inventory</h4><p>${esc(inv)}</p>
    ${a.tools.length ? `<h4>tools</h4><p>${a.tools.join(", ")}</p>` : ""}
    <h4>intention</h4><p>${esc(a.intention) || `<span class="dim">(none)</span>`}</p>
    <h4>plan queue</h4>${plan}
    <h4>knows</h4><p>${a.knowledge.join(", ") || `<span class="dim">no confirmed recipes</span>`}</p>
    ${a.heard.length ? `<h4>heard of</h4><p>${a.heard.join(", ")}</p>` : ""}
    <h4>beliefs</h4>${beliefs}
    <h4>notes (scratchpad)</h4><pre>${esc(a.notes || "(empty)")}</pre>
    <details><summary>journal — what ${a.id} has seen</summary>
      <pre>${esc(a.journal.join("\n") || "(empty)")}</pre></details>
    <h4>last llm call</h4>${llm}`;
}

function fmtStep(s) {
  switch (s.action) {
    case "move_to": return `move_to (${s.x}, ${s.y})`;
    case "gather": return `gather ${s.resource}`;
    case "eat": return `eat ${s.item}`;
    case "combine": return `combine ${Object.entries(s.items).map(([k, v]) => `${v} ${k}`).join(" + ")}`;
    case "say": return `say → ${s.target || "all"}: “${s.message}”`;
    case "give": return `give ${s.qty} ${s.item} → ${s.target}`;
    default: return Object.values(s).join(" ");
  }
}

/* ------------------------------------------------------------- controls */
$("btnMock").onclick = () =>
  post("/api/start", { provider: "mock", condition: "blind", agents: 10,
                       ticks: 300, seed: 42, tick_delay: 0.12 });
$("btnAcceptance").onclick = () =>
  post("/api/start", { provider: "ollama", model: "gemma4:e2b", condition: "blind",
                       agents: 10, ticks: 1000, seed: 42,
                       name: "m3_accept_ui_" + new Date().toISOString().slice(11, 19).replaceAll(":", "") });
$("btnCustom").onclick = () =>
  post("/api/start", {
    provider: $("fProvider").value, model: $("fModel").value || null,
    condition: $("fCondition").value, agents: +$("fAgents").value,
    ticks: +$("fTicks").value, seed: +$("fSeed").value,
    name: $("fName").value || null,
  });
$("btnPause").onclick = () => post("/api/pause");
$("btnResume").onclick = () => post("/api/resume");
$("btnStop").onclick = () => post("/api/stop");
$("speed").oninput = (e) => {
  $("speedVal").textContent = e.target.value + " ms";
  post("/api/speed", { tick_delay: e.target.value / 1000 });
};

/* ------------------------------------------------------------- legend */
(function legend() {
  const items = [["grass", SPR.grass], ["berries", SPR.bush], ["forest", SPR.tree],
    ["stone", SPR.rock], ["water", SPR.water0], ["fire", SPR.fire0],
    ["farm", SPR.farm], ["storehouse", SPR.storehouse],
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

poll();
draw();
