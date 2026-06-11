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
               "ok": err is None, "error": err,
               "system": system, "prompt": user, "response": raw}
        with self._store_lock:
            self._store[obs.me["id"]] = rec


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

    # ---------- event accounting (called from worker thread via EventTee) ----
    def consume_event(self, ev: dict):
        c = self.counters
        t = ev["type"]
        if t == "planner_called":
            c["planner_calls"] = c.get("planner_calls", 0) + 1
        elif t == "planner_parse_error":
            c["parse_errors"] = c.get("parse_errors", 0) + 1
        elif t == "agent_died":
            c["deaths"] = c.get("deaths", 0) + 1
        elif t == "recipe_learned":
            r = ev["payload"]["recipe"]
            c.setdefault("discovered", {}).setdefault(r, ev["tick"])
            if ev["payload"].get("source") == "taught":
                c["taught"] = c.get("taught", 0) + 1
        if t not in ("resource_regrown",):     # regrowth spams the feed
            self.events.append(ev)

    # ---------- control ------------------------------------------------------
    def start(self, params: dict) -> dict:
        with self.lock:
            if self.status in ("running", "paused"):
                raise RuntimeError("a run is already active — stop it first")
            provider = params.get("provider", "mock")
            if provider not in ("mock", "ollama", "anthropic"):
                raise ValueError(f"unknown provider {provider!r}")
            condition = params.get("condition", "blind")
            n_agents = int(params.get("agents", 10))
            ticks = int(params.get("ticks", 300))
            seed = int(params.get("seed", 42))
            model = params.get("model") or None
            name = params.get("name") or (
                f"ui_{provider}_{condition}_s{seed}_{time.strftime('%H%M%S')}")
            self.tick_delay = float(params.get("tick_delay", 0.12 if provider == "mock" else 0.0))

            cfg = dict(BASE, seed=seed, n_agents=n_agents,
                       knowledge_condition=condition,
                       replan_interval=int(params.get("replan", 15)))
            run_dir = RUNS / name
            run_dir.mkdir(parents=True, exist_ok=True)
            telemetry = run_dir / "prompts.jsonl"
            telemetry.write_text("", encoding="utf-8")

            spec = {"backend": provider}
            if model:
                spec["model"] = model

            self.prompt_store.clear()
            self.events.clear()
            self.counters = {}
            self.error = None

            def factory(rng):
                return InspectableBrain(make_provider(spec, rng),
                                        telemetry_path=str(telemetry),
                                        store=self.prompt_store,
                                        store_lock=self.prompt_lock)

            sim = Simulation(cfg, factory, str(run_dir))
            sim.logf = EventTee(sim.logf, self)
            with open(run_dir / "config.json", "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2)

            self.sim = sim
            self.run_info = {"name": name, "provider": provider,
                             "model": model or ("—" if provider == "mock" else "default"),
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
