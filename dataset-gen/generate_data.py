"""Generate random input -> output pairs without playing, all into one file.

    python generate_data.py [-n COUNT] [--level FILE] [--level-share FRAC] [--maps M]
                            [--grounded FRAC] [--seed SEED]

A share of pairs (--level-share, default 40%) use a real level (--level, default levels/level2.txt).
The rest are spread over random maps made by map_gen.py, each with one random spawn
point S on a clear spot. Each file holds --maps levels in total (default 100), --level included.

Each pair: a random frame 1 (position, velocity, keys) and frame 2 computed by the real physics,
in the same format as gamerecorder.py (frame 2 is the spawn point after a death; isDead and
isFinished belong to frame 1), except that LEVEL is an index into the file's "levels" list instead
of the level text. Frame 1 is always a state real play can be in: never inside a wall, and never
touching spikes or the exit (touching those already ended the run). A fraction of samples
(--grounded) stand exactly on a floor with vy = 0, since random positions almost never land on the
ground and jumping / ground friction would otherwise never appear.

Output: generated/<timestamp>.json, minified: {"levels": [text, ...], "pairs": [...]}. Each level's
text is stored once at the top; pairs reference it by index. levels[0] is --level (when
--level-share > 0), the rest are the random maps.
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

from map_gen import make_maps_np  # noqa: E402

from gamerecorder import make_pair
from platformnet.level import LEVEL_PATH, load_level, parse_level, resolve_level
from platformnet.physics import EMPTY, EXIT, GRID_H, GRID_W, JUMP_VEL, MAX_FALL, MAX_RUN, PLAYER, SOLID, SPIKE, TILE
from platformnet.run import advance

GENERATED_DIR = Path(__file__).resolve().parent / "generated"
DEFAULT_LEVEL_PATH = LEVEL_PATH
MAX_X = GRID_W * TILE - PLAYER
MAX_Y = GRID_H * TILE - PLAYER
SIZE = PLAYER // TILE  # player is SIZE x SIZE tiles
TRIES = 1000  # placement attempts before giving up on a map
TILE_TO_CHAR = {EMPTY: ".", SOLID: "#", EXIT: "E", SPIKE: "X"}


def is_reachable(state, level):
    """Alive and still playing: not in a wall, not touching spikes (you'd have died) or the exit (you'd be done)."""
    return not any(level.overlaps(state, t) for t in (SOLID, SPIKE, EXIT))


def random_levels(count, rng):
    """count map_gen maps, each with one spawn S on a clear SIZE x SIZE spot (maps with none are redrawn).
    Returns [(level text, Level), ...]."""
    levels = []
    while len(levels) < count:
        for tiles in make_maps_np(count - len(levels), seed=rng.randrange(2**32)):
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


def generate(count, level_path=DEFAULT_LEVEL_PATH, level_share=0.4, maps=100, grounded=0.5, seed=None,
             make_levels=random_levels):
    """Each pair uses level_path with chance level_share, otherwise one of the random maps;
    `maps` levels in total, level_path included. level_path may also be a list of paths, in which
    case the level_share pairs are split evenly between them. Returns (level texts, pairs); each
    pair's LEVEL is an index into the level texts. make_levels(count, rng) makes the random maps."""
    rng = random.Random(seed)
    paths = [level_path] if isinstance(level_path, (str, Path)) else list(level_path)
    paths = [resolve_level(p) for p in paths]
    fixed = [(p.read_text(), load_level(p)) for p in paths] if level_share > 0 else []
    randoms = make_levels(maps - len(fixed), rng) if level_share < 1 else []
    levels = fixed + randoms
    pairs = []
    fixed_misses = 0
    while len(pairs) < count:
        use_fixed = rng.random() < level_share
        index = rng.randrange(len(fixed)) if use_fixed else len(fixed) + rng.randrange(len(randoms))
        level = levels[index][1]
        if rng.random() < grounded:
            state = random_grounded_state(level, rng)
        else:
            state = random_air_state(level, rng)
        if state is None:  # no valid spot found this time (rare); draw again
            fixed_misses = fixed_misses + 1 if use_fixed else fixed_misses
            if fixed_misses > 100:
                raise ValueError(f"no valid start position found in {paths[index]}")
            continue
        action = (rng.choice((-1, 0, 1)), rng.choice((0, 1)))
        _, dead, new_state, finished = advance(state, action, level)
        pairs.append(make_pair(index, state, action, new_state, dead, finished))
    return [text for text, _ in levels], pairs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-n", "--count", type=int, default=250000, help="number of pairs (default 5000)")
    parser.add_argument("--level", default=DEFAULT_LEVEL_PATH, help="real level mixed in (default levels/level2.txt)")
    parser.add_argument("--level-share", type=float, default=0.4,
                        help="fraction of pairs on --level; the rest get random maps (default 0.4; 1 = all, 0 = none)")
    parser.add_argument("--maps", type=int, default=100, help="levels per file, --level included (default 100)")
    parser.add_argument("--grounded", type=float, default=0.5, help="fraction starting on a floor (default 0.5)")
    parser.add_argument("--seed", type=int, help="random seed, for repeatable output")
    args = parser.parse_args()
    if not 0 <= args.level_share <= 1:
        parser.error("--level-share must be between 0 and 1")
    if 0 < args.level_share < 1 and args.maps < 2:
        parser.error("--maps must be at least 2 when mixing --level with random maps")
    if args.level_share == 0 and args.maps < 1:
        parser.error("--maps must be at least 1")

    levels, pairs = generate(args.count, args.level, args.level_share, args.maps, args.grounded, args.seed)
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    path = GENERATED_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(json.dumps({"levels": levels, "pairs": pairs}, separators=(",", ":")))
    print(f"Generated {len(pairs)} pairs on {len(levels)} levels to {path}")
