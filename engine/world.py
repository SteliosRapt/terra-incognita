"""world.py — the grid, tiles, resources, and their dynamics."""
from __future__ import annotations
import random
from dataclasses import dataclass, field
from .events import Event

GRASS, FOREST, MOUNTAIN, DIRT, WATER = "grass", "forest", "mountain", "dirt", "water"


@dataclass
class Tile:
    type: str
    # resource state
    wood: int = 0                 # forest: 1 when tree present
    wood_regrow_at: int = -1      # tick when tree returns
    stone: int = 0                # mountain: finite stock
    berries: int = 0              # grass: 0/1, stochastic regrowth
    berry_capable: bool = False   # only patch tiles regrow berries

    @property
    def passable(self) -> bool:
        return self.type != WATER


@dataclass
class Structure:
    type: str                    # "fire" | "farm" | "storehouse"
    owner: str
    created: int
    expires_at: int = -1         # -1 = permanent
    storage: dict = field(default_factory=dict)   # storehouse contents
    STORE_UNITS = 200            # 40 slots x 5 units

    def units(self) -> int:
        return sum(self.storage.values())


class World:
    def __init__(self, cfg: dict, rng: random.Random):
        self.cfg = cfg
        self.rng = rng
        self.W = cfg["grid_size"]
        self.H = cfg["grid_size"]
        self.grid: list[list[Tile]] = []
        self.structures: dict[tuple[int, int], "Structure"] = {}
        self.techtree = None      # attached by Simulation
        self._generate()

    # ---------- generation ----------
    def _grow_region(self, budget: int, convert):
        """Organic region growth: seed on grass, repeatedly convert a random
        grass neighbor of the region. Produces connected random shapes."""
        for _ in range(200):  # find a grass seed
            sx, sy = self.rng.randrange(self.W), self.rng.randrange(self.H)
            if self.grid[sy][sx].type == GRASS:
                break
        else:
            return
        convert(self.grid[sy][sx])
        region = [(sx, sy)]
        attempts = 0
        while len(region) < budget and attempts < budget * 30:
            attempts += 1
            x, y = region[self.rng.randrange(len(region))]
            nx = (x + self.rng.choice((-1, 0, 1))) % self.W
            ny = (y + self.rng.choice((-1, 0, 1))) % self.H
            t = self.grid[ny][nx]
            if t.type == GRASS:
                convert(t)
                region.append((nx, ny))

    def _generate(self):
        c = self.cfg
        self.grid = [[Tile(GRASS) for _ in range(self.W)] for _ in range(self.H)]
        area = self.W * self.H

        def make(kind):
            def f(t: Tile):
                t.type = kind
                if kind == FOREST:
                    t.wood = 1
                elif kind == MOUNTAIN:
                    t.stone = c["stone_per_tile"]
            return f

        # lakes first (so forests/mountains shape around them)
        n_lakes = c.get("lakes", 3)
        for _ in range(n_lakes):
            self._grow_region(int(area * c.get("water_fraction", 0.10) / n_lakes),
                              make(WATER))
        # forests: few large connected regions
        n_forests = c.get("forest_regions", 4)
        for _ in range(n_forests):
            self._grow_region(int(area * c["forest_fraction"] / n_forests),
                              make(FOREST))
        # mountain ranges
        n_ranges = c.get("mountain_ranges", 3)
        for _ in range(n_ranges):
            self._grow_region(int(area * c["mountain_fraction"] / n_ranges),
                              make(MOUNTAIN))
        # berry patches (grass only, never water)
        for _ in range(c["berry_patches"]):
            for _ in range(200):
                cx, cy = self.rng.randrange(self.W), self.rng.randrange(self.H)
                if self.grid[cy][cx].type == GRASS:
                    break
            for _ in range(c["berry_patch_size"]):
                x = (cx + self.rng.randint(-2, 2)) % self.W
                y = (cy + self.rng.randint(-2, 2)) % self.H
                t = self.tile(x, y)
                if t.type == GRASS:
                    t.berry_capable = True
                    t.berries = 1

    # ---------- helpers ----------
    def tile(self, x: int, y: int) -> Tile:
        return self.grid[y % self.H][x % self.W]

    def torus_dist(self, a: tuple[int, int], b: tuple[int, int]) -> int:
        dx = min(abs(a[0] - b[0]), self.W - abs(a[0] - b[0]))
        dy = min(abs(a[1] - b[1]), self.H - abs(a[1] - b[1]))
        return max(dx, dy)  # Chebyshev: 8-directional movement

    def next_step(self, a: tuple[int, int], b: tuple[int, int]):
        """BFS next step from a toward b over passable tiles (8-dir, toroidal).
        Deterministic neighbor order. Returns None if unreachable or b impassable."""
        if a == b:
            return a
        if not self.tile(*b).passable:
            return None
        from collections import deque
        prev = {a: None}
        q = deque([a])
        DIRS = ((0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1))
        while q:
            cur = q.popleft()
            if cur == b:
                break
            for dx, dy in DIRS:
                nxt = ((cur[0] + dx) % self.W, (cur[1] + dy) % self.H)
                if nxt not in prev and self.tile(*nxt).passable:
                    prev[nxt] = cur
                    q.append(nxt)
        if b not in prev:
            return None
        node = b
        while prev[node] != a:
            node = prev[node]
        return node

    def find_tiles(self, pred, center: tuple[int, int], radius: int):
        """Tiles within Chebyshev radius of center satisfying pred, sorted by distance.
        Deterministic order: distance, then (x, y)."""
        out = []
        cx, cy = center
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                x, y = (cx + dx) % self.W, (cy + dy) % self.H
                if pred(self.tile(x, y)):
                    out.append(((x, y), self.torus_dist(center, (x, y))))
        out.sort(key=lambda e: (e[1], e[0]))
        return [pos for pos, _ in out]

    def adjacent_structures(self, pos: tuple[int, int]) -> set:
        """Structure types on own tile or Chebyshev-adjacent tiles."""
        out = set()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                p = ((pos[0] + dx) % self.W, (pos[1] + dy) % self.H)
                if p in self.structures:
                    out.add(self.structures[p].type)
        return out

    # ---------- world phase ----------
    def world_phase(self, tick: int) -> list[Event]:
        evs = []
        c = self.cfg
        # structures: fire expiry, farm yields
        for pos in sorted(self.structures.keys()):
            st = self.structures[pos]
            if st.expires_at == tick:
                del self.structures[pos]
                evs.append(Event(tick, "structure_expired", position=pos,
                                 payload={"structure": st.type}))
                continue
            if st.type == "farm" and tick > st.created and (tick - st.created) % c.get("farm_yield_every", 10) == 0:
                t = self.tile(*pos)
                t.berries = min(t.berries + c.get("farm_yield_amount", 3), 9)
                evs.append(Event(tick, "farm_yielded", position=pos,
                                 payload={"item": "berries",
                                          "amount": c.get("farm_yield_amount", 3)}))
        for y in range(self.H):
            for x in range(self.W):
                t = self.grid[y][x]
                if t.type == FOREST and t.wood == 0 and t.wood_regrow_at == tick:
                    t.wood = 1
                    evs.append(Event(tick, "resource_regrown", position=(x, y),
                                     payload={"resource": "wood"}))
                if (t.berry_capable and t.berries == 0
                        and self.rng.random() < c["berry_regrow_prob"]):
                    t.berries = 1
                    evs.append(Event(tick, "resource_regrown", position=(x, y),
                                     payload={"resource": "berries"}))
        return evs
