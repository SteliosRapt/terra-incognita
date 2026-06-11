"""viewer/server.py — live dashboard for terra incognita.

Zero new dependencies: stdlib http.server + threading. The simulation runs in
a worker thread inside this process; the browser polls /api/state.

The engine is untouched: events are captured by teeing the sim's log file
object, and agent prompts/responses by subclassing LLMPlannerBrain._log.
Beliefs/notes are read here for *display only* — the viewer is measurement,
like the event log, never world logic.

Usage:  py viewer/server.py [--port 8787] [--no-open]
"""
from __future__ import annotations
import argparse
import collections
import json
import pathlib
import sys
import threading
import time
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine.sim import Simulation                      # noqa: E402
from brains.llm_planner import LLMPlannerBrain         # noqa: E402
from brains.providers import make_provider             # noqa: E402
from run_m3 import BASE, ALL_RECIPES                   # noqa: E402

STATIC = pathlib.Path(__file__).resolve().parent / "static"
RUNS = ROOT / "runs"

TERRAIN_CHAR = {"grass": "g", "forest": "T", "mountain": "M", "water": "w"}

# world keys the UI may override at /api/start (defaults-first: BASE fills the form)
WORLD_INT = {"grid_size", "stone_per_tile", "wood_regrow_ticks", "berry_patches",
             "berry_patch_size", "lakes", "forest_regions", "mountain_ranges"}
WORLD_FLOAT = {"forest_fraction", "mountain_fraction", "berry_regrow_prob",
               "berry_spoil_prob", "water_fraction"}

# model presets (the configurator's quick options; "custom" uses provider/model)
PRESETS = {
    "local-e2b": "ollama gemma4:e2b, all agents",
    "openrouter-90-10": "90% gemma-4 free / 10% (floor) qwen3.6-35b-a3b paid",
    "mock": "deterministic mock brain",
    "custom": "provider + model fields",
}


class EventTee:
    """Wraps the sim's events.jsonl file object: still writes (and now flushes)
    every line, but also feeds a ring buffer + counters for the dashboard."""

    def __init__(self, f, manager: "RunManager"):
        self.f = f
        self.m = manager

    def write(self, s: str):
        self.f.write(s)
        self.f.flush()
        line = s.strip()
        if line:
            self.m.consume_event(json.loads(line))

    def flush(self):
        self.f.flush()

    def close(self):
        self.f.close()


class InspectableBrain(LLMPlannerBrain):
    """Same planner, but the dashboard keeps each agent's latest exchange."""

    def __init__(self, *args, store=None, store_lock=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._store = store
        self._store_lock = store_lock

    def _log(self, obs, attempt, system, user, raw, ms, err):
        super()._log(obs, attempt, system, user, raw, ms, err)
        if self._store is None:
            return
        rec = {"tick": obs.tick, "attempt": attempt, "ms": ms,
               "ok": err is None, "error": err, "model": self.provider.name,
               "system": system, "prompt": user, "response": raw}
        with self._store_lock:
            self._store[obs.me["id"]] = rec


def compute_baseline() -> dict | None:
    """Reference stats from the no-LLM runs already in runs/: scripted M2 runs
    (no prompts.jsonl) and MockProvider runs (telemetry says provider=mock).
    Tick-weighted totals, normalized — the dashboard shows the live LLM run
    against these numbers."""
    runs_used, ticks = [], 0
    deaths = calls = completed = gathered = 0
    recipes = set()
    for d in sorted(RUNS.iterdir()) if RUNS.exists() else []:
        ev_path = d / "events.jsonl"
        if not d.is_dir() or not ev_path.exists():
            continue
        p = d / "prompts.jsonl"
        if p.exists() and p.stat().st_size:
            try:
                with open(p, encoding="utf-8") as f:
                    first = json.loads(f.readline())
                if first.get("provider") != "mock":
                    continue                       # an LLM run — not a baseline
            except (json.JSONDecodeError, OSError):
                continue
        try:
            run_ticks = 0
            with open(ev_path, encoding="utf-8") as f:
                for line in f:
                    ev = json.loads(line)
                    run_ticks = max(run_ticks, ev["tick"] + 1)
                    t = ev["type"]
                    if t == "agent_died":
                        deaths += 1
                    elif t == "planner_called":
                        calls += 1
                    elif t == "plan_completed":
                        completed += 1
                    elif t == "gathered":
                        gathered += 1
                    elif t == "recipe_learned":
                        recipes.add(ev["payload"]["recipe"])
        except (json.JSONDecodeError, OSError, KeyError):
            continue
        if run_ticks:
            runs_used.append(d.name)
            ticks += run_ticks
    if not ticks:
        return None
    return {"runs": runs_used, "ticks": ticks,
            "deaths_per_1k": round(deaths / ticks * 1000, 2),
            "completion_pct": round(completed / calls * 100, 1) if calls else 0.0,
            "gathers_per_100": round(gathered / ticks * 100, 1),
            "recipes_discovered": len(recipes)}


class RunManager:
    """Owns at most one live simulation and a snapshot the API can serve
    without blocking on a tick in progress (LLM ticks take seconds)."""

    def __init__(self):
        self.lock = threading.Lock()          # guards snapshot + status fields
        self.sim: Simulation | None = None
        self.thread: threading.Thread | None = None
        self.status = "idle"                  # idle|running|paused|finished|stopped|error
        self.error: str | None = None
        self.run_info: dict = {}
        self.snapshot: dict | None = None
        self.tick_delay = 0.0
        self._stop = threading.Event()
        self._pause = threading.Event()
        # live-derived measurements
        self.events = collections.deque(maxlen=400)
        self.counters: dict = {}
        self.prompt_store: dict = {}
        self.prompt_lock = threading.Lock()
        self.providers: list = []             # shared LLM provider(s) of the live run
        self.baseline = compute_baseline()    # no-LLM reference stats (runs/ scan)

    # ---------- event accounting (called from worker thread via EventTee) ----
    def consume_event(self, ev: dict):
        c = self.counters
        t = ev["type"]
        if t == "planner_called":
            c["planner_calls"] = c.get("planner_calls", 0) + 1
        elif t == "planner_parse_error":
            c["parse_errors"] = c.get("parse_errors", 0) + 1
        elif t == "plan_completed":
            c["plans_completed"] = c.get("plans_completed", 0) + 1
        elif t == "gathered":
            c["gathered"] = c.get("gathered", 0) + 1
        elif t == "agent_died":
            c["deaths"] = c.get("deaths", 0) + 1
        elif t == "recipe_learned":
            r = ev["payload"]["recipe"]
            c.setdefault("discovered", {}).setdefault(
                r, {"tick": ev["tick"], "by": ev["actor"],
                    "source": ev["payload"].get("source", "discovered")})
            if ev["payload"].get("source") == "taught":
                c["taught"] = c.get("taught", 0) + 1
        if t not in ("resource_regrown",):     # regrowth spams the feed
            self.events.append(ev)

    # ---------- control ------------------------------------------------------
    def start(self, params: dict) -> dict:
        with self.lock:
            if self.status in ("running", "paused"):
                raise RuntimeError("a run is already active — stop it first")
            preset = params.get("preset", "custom")
            if preset not in PRESETS:
                raise ValueError(f"unknown preset {preset!r}")
            condition = params.get("condition", "blind")
            n_agents = int(params.get("agents", 10))
            ticks = int(params.get("ticks", 300))
            seed = int(params.get("seed", 42))
            model = params.get("model") or None
            if preset == "local-e2b":
                provider, model = "ollama", "gemma4:e2b"
            elif preset == "mock":
                provider, model = "mock", None
            elif preset == "openrouter-90-10":
                provider, model = "openrouter", None
            else:
                provider = params.get("provider", "mock")
                if provider not in ("mock", "ollama", "anthropic", "openrouter"):
                    raise ValueError(f"unknown provider {provider!r}")
            name = params.get("name") or (
                f"ui_{preset if preset != 'custom' else provider}"
                f"_{condition}_s{seed}_{time.strftime('%H%M%S')}")
            self.tick_delay = float(params.get("tick_delay", 0.12 if provider == "mock" else 0.0))

            cfg = dict(BASE, seed=seed, n_agents=n_agents,
                       knowledge_condition=condition,
                       replan_interval=int(params.get("replan", 15)))
            for k, v in (params.get("world") or {}).items():
                if k in WORLD_INT:
                    cfg[k] = int(v)
                elif k in WORLD_FLOAT:
                    cfg[k] = float(v)
                else:
                    raise ValueError(f"unknown world parameter {k!r}")
            run_dir = RUNS / name
            run_dir.mkdir(parents=True, exist_ok=True)
            telemetry = run_dir / "prompts.jsonl"
            telemetry.write_text("", encoding="utf-8")

            self.prompt_store.clear()
            self.events.clear()
            self.counters = {}
            self.error = None
            self.providers = []   # shared instances: global cooldowns + spend

            def brain(prov):
                return InspectableBrain(prov, telemetry_path=str(telemetry),
                                        store=self.prompt_store,
                                        store_lock=self.prompt_lock)

            paid_agents: list[str] = []
            if preset == "openrouter-90-10":
                # floor(10%) of the population on the paid qwen3.6 chain, the
                # rest on free gemma-4. Providers are built NOW so a missing
                # OPENROUTER_API_KEY fails the start request, not tick 1.
                cost_path = str(run_dir / "costs.jsonl")
                free = make_provider({"backend": "openrouter",
                                      "cost_path": cost_path})
                paid = make_provider({"backend": "openrouter", "tier": "paid",
                                      "cost_path": cost_path})
                self.providers = [free, paid]
                n_paid = n_agents // 10
                paid_agents = [f"A{i:02d}" for i in range(n_paid)]
                counter = iter(range(n_agents))   # sim builds brains in name order

                def factory(rng):
                    return brain(paid if next(counter) < n_paid else free)
                model_label = (f"{free.models[0]} ({n_agents - n_paid}) + "
                               f"{paid.models[0]} ({n_paid}: {', '.join(paid_agents) or '—'})")
            else:
                spec = {"backend": provider}
                if model:
                    spec["model"] = model
                if provider == "openrouter":
                    spec["cost_path"] = str(run_dir / "costs.jsonl")

                def factory(rng):
                    if not self.providers:
                        self.providers = [make_provider(spec, rng)]
                    return brain(self.providers[0])
                model_label = model or ("—" if provider == "mock" else "default")

            sim = Simulation(cfg, factory, str(run_dir))
            sim.logf = EventTee(sim.logf, self)
            with open(run_dir / "config.json", "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2)

            self.sim = sim
            self.run_info = {"name": name, "provider": provider,
                             "preset": preset, "model": model_label,
                             "paid_agents": paid_agents,
                             "condition": condition, "agents": n_agents,
                             "ticks": ticks, "seed": seed,
                             "started": time.time()}
            self.status = "running"
            self._stop.clear()
            self._pause.clear()
            self.snapshot = self._build_snapshot(sim)
            self.thread = threading.Thread(target=self._worker, args=(sim, ticks),
                                           daemon=True)
            self.thread.start()
            return dict(self.run_info)

    def _worker(self, sim: Simulation, ticks: int):
        try:
            while sim.tick < ticks and not self._stop.is_set():
                if self._pause.is_set():
                    time.sleep(0.1)
                    continue
                sim.step()
                snap = self._build_snapshot(sim)
                with self.lock:
                    self.snapshot = snap
                if not any(a.alive for a in sim.agents.values()):
                    break
                if self.tick_delay:
                    time.sleep(self.tick_delay)
            with self.lock:
                self.status = "stopped" if self._stop.is_set() else "finished"
        except Exception:
            with self.lock:
                self.status = "error"
                self.error = traceback.format_exc(limit=8)
        finally:
            try:
                sim.logf.close()
            except Exception:
                pass

    def pause(self):
        with self.lock:
            if self.status == "running":
                self._pause.set()
                self.status = "paused"

    def resume(self):
        with self.lock:
            if self.status == "paused":
                self._pause.clear()
                self.status = "running"

    def stop(self):
        self._stop.set()
        self._pause.clear()

    def set_speed(self, delay: float):
        self.tick_delay = max(0.0, min(2.0, delay))

    # ---------- snapshot -----------------------------------------------------
    def _build_snapshot(self, sim: Simulation) -> dict:
        w = sim.world
        rows = []
        for y in range(w.H):
            row = []
            for x in range(w.W):
                t = w.grid[y][x]
                ch = TERRAIN_CHAR.get(t.type, "g")
                if t.type == "grass" and t.berries > 0:
                    ch = "b"
                elif t.type == "forest" and t.wood == 0:
                    ch = "t"
                elif t.type == "mountain" and t.stone == 0:
                    ch = "m"
                row.append(ch)
            rows.append("".join(row))
        structures = [{"type": s.type, "x": p[0], "y": p[1], "owner": s.owner,
                       "expires_at": s.expires_at,
                       "storage": dict(s.storage)}
                      for p, s in sorted(w.structures.items())]
        agents = []
        for name in sorted(sim.agents):
            a = sim.agents[name]
            agents.append({
                "id": a.id, "x": a.x, "y": a.y, "alive": a.alive,
                "energy": a.energy, "inventory": dict(a.inventory),
                "tools": sorted(a.tools), "knowledge": sorted(a.knowledge),
                "heard": sorted(a.heard_recipes),
                "intention": a.intention, "plan": list(a.plan),
                "beliefs": list(a.beliefs), "notes": a.notes,
                "journal": list(a.journal[-12:]),
                "last_plan_tick": a.last_plan_tick,
            })
        c = self.counters
        calls = c.get("planner_calls", 0)
        errs = c.get("parse_errors", 0)
        spend = None
        for p in self.providers:
            t = getattr(p, "totals", None)
            if t:
                spend = ({k: spend.get(k, 0) + v for k, v in t.items()}
                         if spend else dict(t))
        ticks_done = max(sim.tick, 1)
        return {
            "tick": sim.tick,
            "grid": rows, "W": w.W, "H": w.H,
            "vis_radius": sim.vis_radius,
            "structures": structures,
            "agents": agents,
            "stats": {
                "alive": sum(a.alive for a in sim.agents.values()),
                "deaths": c.get("deaths", 0),
                "planner_calls": calls,
                "parse_errors": errs,
                "parse_rate": round(errs / calls * 100, 1) if calls else 0.0,
                "taught": c.get("taught", 0),
                "discovered": c.get("discovered", {}),
                "recipes": ALL_RECIPES,
                "spend": spend,
                "live": {   # normalized like the baseline, for the comparison panel
                    "deaths_per_1k": round(c.get("deaths", 0) / ticks_done * 1000, 2),
                    "completion_pct": round(c.get("plans_completed", 0) / calls * 100, 1) if calls else 0.0,
                    "gathers_per_100": round(c.get("gathered", 0) / ticks_done * 100, 1),
                    "recipes_discovered": len(c.get("discovered", {})),
                },
            },
            "events": list(self.events)[-60:],
        }

    # ---------- state for the API -------------------------------------------
    def state(self) -> dict:
        with self.lock:
            out = {
                "status": self.status,
                "error": self.error,
                "run": dict(self.run_info),
                "tick_delay": self.tick_delay,
                "snapshot": self.snapshot,
            }
        with self.prompt_lock:
            out["prompts"] = dict(self.prompt_store)
        out["external"] = self._external_runs()
        out["baseline"] = self.baseline
        return out

    def _external_runs(self) -> list:
        """Other processes writing into runs/ right now (e.g. a CLI acceptance
        run) — surfaced so the user doesn't double-load Ollama unknowingly."""
        own = self.run_info.get("name")
        active = []
        now = time.time()
        try:
            for d in RUNS.iterdir():
                if not d.is_dir() or d.name == own:
                    continue
                p = d / "prompts.jsonl"
                if p.exists() and now - p.stat().st_mtime < 180:
                    active.append(d.name)
        except OSError:
            pass
        return sorted(active)


MANAGER = RunManager()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):       # quiet: one line per poll is noise
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _file(self, path: pathlib.Path):
        types = {".html": "text/html", ".css": "text/css",
                 ".js": "text/javascript", ".png": "image/png"}
        try:
            data = path.read_bytes()
        except OSError:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type",
                         types.get(path.suffix, "application/octet-stream") + "; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/api/state":
            self._json(MANAGER.state())
        elif path == "/api/defaults":
            self._json({"world": {k: BASE[k] for k in
                                  sorted(WORLD_INT | WORLD_FLOAT)},
                        "replan_interval": BASE["replan_interval"],
                        "presets": PRESETS})
        elif path == "/":
            self._file(STATIC / "index.html")
        else:
            p = (STATIC / path.lstrip("/")).resolve()
            if STATIC.resolve() in p.parents and p.is_file():
                self._file(p)
            else:
                self.send_error(404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        try:
            params = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._json({"error": "bad JSON body"}, 400)
            return
        try:
            if self.path == "/api/start":
                self._json({"ok": True, "run": MANAGER.start(params)})
            elif self.path == "/api/pause":
                MANAGER.pause()
                self._json({"ok": True})
            elif self.path == "/api/resume":
                MANAGER.resume()
                self._json({"ok": True})
            elif self.path == "/api/stop":
                MANAGER.stop()
                self._json({"ok": True})
            elif self.path == "/api/speed":
                MANAGER.set_speed(float(params.get("tick_delay", 0)))
                self._json({"ok": True})
            else:
                self.send_error(404)
        except (RuntimeError, ValueError) as e:
            self._json({"error": str(e)}, 409)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8787)
    ap.add_argument("--no-open", action="store_true")
    args = ap.parse_args()
    url = f"http://127.0.0.1:{args.port}/"
    try:
        srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    except OSError:
        # already running (e.g. shortcut double-clicked twice) — just open it
        print(f"port {args.port} busy — opening the existing field station")
        webbrowser.open(url)
        return
    print(f"terra incognita field station: {url}")
    if not args.no_open:
        webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        MANAGER.stop()


if __name__ == "__main__":
    main()
