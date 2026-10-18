"""Like generate_data.py, but the random maps are clumpy caves instead of map_gen maps.

    python generate_clumped_data.py [-n COUNT] [--level FILE] [--level-share FRAC] [--maps M]
                                    [--grounded FRAC] [--seed SEED]

Maps come from cellular-automaton smoothing: start from random noise, then repeatedly make each
tile solid if most of its 8 neighbours are solid and empty if most are empty. Lone tiles die out,
so solid tiles clump into walls and masses, and empty tiles open up into air pockets and tunnels.
Each map draws its own fill density, smoothing steps and feature scale (some are built on a
coarser grid and upscaled, giving bigger rooms and thicker walls). Spikes are then laid in short
runs on floors, and an exit clump goes in an open pocket, so all tile types appear. Each map gets
one random spawn point S on a clear spot.

Everything else matches generate_data.py: a share of pairs (--level-share) use a real level, each
file holds --maps levels in total (default 100, --level included), frame 1 is always a reachable
state, and --grounded of the samples stand exactly on a floor.

Output: generated/<timestamp>-clumped.json, minified: {"levels": [text, ...], "pairs": [...]},
with each pair's LEVEL an index into "levels".
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np

from gamerecorder import make_pair
from platformnet.level import load_level, parse_level, resolve_level
from platformnet.physics import (EMPTY, EXIT, GRID_H, GRID_W, JUMP_VEL, MAX_FALL, MAX_RUN, PLAYER, SOLID,
                                 SPIKE, TILE)
from platformnet.run import advance

GENERATED_DIR = Path(__file__).resolve().parent / "generated"
MAX_X = GRID_W * TILE - PLAYER
MAX_Y = GRID_H * TILE - PLAYER
SIZE = PLAYER // TILE  # player is SIZE x SIZE tiles
TRIES = 1000  # placement attempts before giving up on a map
TILE_TO_CHAR = {EMPTY: ".", SOLID: "#", EXIT: "E", SPIKE: "X"}


# ---- clumped map generation --------------------------------------------------

def _solid_neighbours(solid):
    """Number of solid tiles among each tile's 8 neighbours; outside the grid counts as solid."""
    p = np.pad(solid, 1, constant_values=True).astype(np.int8)
    h, w = solid.shape
    return sum(p[1 + dr:1 + dr + h, 1 + dc:1 + dc + w]
               for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc)


def _smooth(solid, steps):
    """Cave rule: more than 4 solid neighbours -> solid, fewer than 4 -> empty, exactly 4 -> unchanged."""
    for _ in range(steps):
        n = _solid_neighbours(solid)
        solid = (n > 4) | (solid & (n == 4))
    return solid


def _caves(rng):
    """GRID_H x GRID_W bool array (True = solid) of clumped walls and air pockets."""
    scale = rng.choice((1, 1, 2, 2, 4))  # >1: build coarse, upscale -> bigger rooms, thicker walls
    h, w = -(-GRID_H // scale), -(-GRID_W // scale)
    fill = rng.uniform(0.38, 0.52)
    solid = _smooth(rng.random((h, w)) < fill, rng.integers(3, 7))
    solid = np.kron(solid, np.ones((scale, scale), dtype=bool))[:GRID_H, :GRID_W]
    if scale > 1:  # roughen the blocky upscaled edges, then round them off again
        flip = rng.random(solid.shape) < 0.08
        solid = _smooth(solid ^ flip, 2)
    return solid


def _add_spikes(m, rng):
    """Short runs of spikes on floors (empty tile with solid below)."""
    rows, cols = np.nonzero((m[:-1] == EMPTY) & (m[1:] == SOLID))
    if not len(rows):
        return
    for i in rng.choice(len(rows), size=min(len(rows), int(rng.integers(0, 14))), replace=False):
        r, c = rows[i], cols[i]
        seg = slice(c, min(c + int(rng.integers(1, 10)), GRID_W))
        ok = (m[r, seg] == EMPTY) & (m[r + 1, seg] == SOLID)
        m[r, seg][ok] = SPIKE


def _add_exit(m, rng):
    """A small exit clump filling the empty tiles of a random box (sometimes none, sometimes two)."""
    for _ in range(rng.choice((0, 1, 1, 1, 2))):
        w, h = int(rng.integers(2, 7)), int(rng.integers(2, 10))
        r, c = int(rng.integers(0, GRID_H - h + 1)), int(rng.integers(0, GRID_W - w + 1))
        box = m[r:r + h, c:c + w]
        box[box == EMPTY] = EXIT


def make_clumped_map(rng):
    """One GRID_H x GRID_W int array of tile codes."""
    m = np.where(_caves(rng), SOLID, EMPTY)
    _add_spikes(m, rng)
    _add_exit(m, rng)
    return m


def clumped_levels(count, rng):
    """count clumped maps, each with one spawn S on a clear SIZE x SIZE spot (maps with none are redrawn).
    Returns [(level text, Level), ...]."""
    np_rng = np.random.default_rng(rng.randrange(2**32))
    levels = []
    while len(levels) < count:
        tiles = make_clumped_map(np_rng)
        empty = tiles == EMPTY
        clear = [(c, r) for r in range(GRID_H - SIZE + 1) for c in range(GRID_W - SIZE + 1)
                 if empty[r:r + SIZE, c:c + SIZE].all()]
        if not clear:
            continue
        grid = [[TILE_TO_CHAR[t] for t in row] for row in tiles.tolist()]
        c, r = rng.choice(clear)
        grid[r][c] = "S"
        text = "\n".join("".join(row) for row in grid) + "\n"
        levels.append((text, parse_level(text)))
    return levels


# ---- sampling pairs (same as generate_data.py) -------------------------------

def is_reachable(state, level):
    """Alive and still playing: not in a wall, not touching spikes (you'd have died) or the exit (you'd be done)."""
    return not any(level.overlaps(state, t) for t in (SOLID, SPIKE, EXIT))


def random_air_state(level, rng):
    for _ in range(TRIES):
        state = (rng.uniform(0, MAX_X), rng.uniform(0, MAX_Y),
                 rng.uniform(-MAX_RUN, MAX_RUN), rng.uniform(JUMP_VEL, MAX_FALL))
        if is_reachable(state, level):
            return state
    return None


def random_grounded_state(level, rng):
    for _ in range(TRIES):
        x = rng.uniform(0, MAX_X)
        floor_row = rng.randrange(SIZE, GRID_H)
        state = (x, float(floor_row * TILE - PLAYER), rng.uniform(-MAX_RUN, MAX_RUN), 0.0)
        under = range(int(x // TILE), int(-(-(x + PLAYER) // TILE)))
        if any(level.tile(c, floor_row) == SOLID for c in under) and is_reachable(state, level):
            return state
    return None


def generate(count, level_path=None, level_share=0.4, maps=100, grounded=0.5, seed=None):
    """Each pair uses level_path with chance level_share, otherwise one of the clumped maps;
    `maps` levels in total, level_path included. Returns (level texts, pairs); each pair's LEVEL
    is an index into the level texts."""
    rng = random.Random(seed)
    fixed = []
    if level_share > 0:
        level_file = resolve_level(level_path)
        fixed = [(level_file.read_text(), load_level(level_file))]
    n_random = maps - len(fixed) if level_share < 1 else 0
    levels = fixed + clumped_levels(n_random, rng)
    pairs = []
    fixed_misses = 0
    while len(pairs) < count:
        use_fixed = bool(fixed) and (not n_random or rng.random() < level_share)
        index = 0 if use_fixed else len(fixed) + rng.randrange(n_random)
        level = levels[index][1]
        if rng.random() < grounded:
            state = random_grounded_state(level, rng)
        else:
            state = random_air_state(level, rng)
        if state is None:  # no valid spot found this time (rare); draw again
            fixed_misses += use_fixed
            if fixed_misses > 100:
                raise ValueError(f"no valid start position found in {level_file}")
            continue
        action = (rng.choice((-1, 0, 1)), rng.choice((0, 1)))
        _, dead, new_state, finished = advance(state, action, level)
        pairs.append(make_pair(index, state, action, new_state, dead, finished))
    return [text for text, _ in levels], pairs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-n", "--count", type=int, default=250000, help="number of pairs (default 250000)")
    parser.add_argument("--level", default=None, help="real level mixed in (default levels/level2.txt)")
    parser.add_argument("--level-share", type=float, default=0.4,
                        help="fraction of pairs on --level; the rest get clumped maps (default 0.4; 1 = all, 0 = none)")
    parser.add_argument("--maps", type=int, default=100, help="levels per file, --level included (default 100)")
    parser.add_argument("--grounded", type=float, default=0.5, help="fraction starting on a floor (default 0.5)")
    parser.add_argument("--seed", type=int, help="random seed, for repeatable output")
    args = parser.parse_args()
    if not 0 <= args.level_share <= 1:
        parser.error("--level-share must be between 0 and 1")
    if 0 < args.level_share < 1 and args.maps < 2:
        parser.error("--maps must be at least 2 when mixing --level with clumped maps")
    if args.level_share == 0 and args.maps < 1:
        parser.error("--maps must be at least 1")

    levels, pairs = generate(args.count, args.level, args.level_share, args.maps, args.grounded, args.seed)
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    path = GENERATED_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}-clumped.json"
    path.write_text(json.dumps({"levels": levels, "pairs": pairs}, separators=(",", ":")))
    print(f"Generated {len(pairs)} pairs on {len(levels)} levels to {path}")
