"""Generate random input -> output pairs without playing, all into one file.

    python generate_data.py [-n COUNT] [--level FILE] [--solid MIN MAX] [--spike P] [--exit P]
                            [--grounded FRAC] [--seed SEED]

Every pair gets its own random map: each of the 100x100 tiles is independently a wall, spike
(--spike), exit (--exit) or empty, plus one random spawn point S. Each map picks its own wall chance
between --solid MIN and MAX (default 2%-20%), so some maps are sparse and some crowded, but air
always far outnumbers walls. That map's text is the pair's LEVEL. Pass --level FILE to use one fixed level for every pair instead.

Each pair: a random frame 1 (position, velocity, keys) and frame 2 computed by the real physics,
in the same format as gamerecorder.py (frame 2 is the spawn point after a death; isDead and
isFinished belong to frame 1). Frame 1 is always a state real play can be in: never inside a wall,
and never touching spikes or the exit (touching those already ended the run). A fraction of samples
(--grounded) stand exactly on a floor with vy = 0, since random positions almost never land on the
ground and jumping / ground friction would otherwise never appear.

Output: generated/<timestamp>.json
"""
import argparse
import json
import random
import time
from pathlib import Path

from gamerecorder import make_pair
from platformnet.level import load_level, parse_level
from platformnet.physics import EXIT, GRID_H, GRID_W, JUMP_VEL, MAX_FALL, MAX_RUN, PLAYER, SOLID, SPIKE, TILE
from platformnet.run import advance

GENERATED_DIR = Path(__file__).resolve().parent / "generated"
MAX_X = GRID_W * TILE - PLAYER
MAX_Y = GRID_H * TILE - PLAYER
SIZE = PLAYER // TILE  # player is SIZE x SIZE tiles
TRIES = 1000  # placement attempts before giving up on a map


def is_reachable(state, level):
    """Alive and still playing: not in a wall, not touching spikes (you'd have died) or the exit (you'd be done)."""
    return not any(level.overlaps(state, t) for t in (SOLID, SPIKE, EXIT))


def random_level(rng, solid=(0.02, 0.2), spike=0.03, exit_=0.01):
    """A map of independent random tiles plus one spawn S on a clear SIZE x SIZE spot.
    The wall chance is drawn uniformly from the solid (min, max) range, once per map.
    Returns (level text, Level), or None if no clear spot was found."""
    solid = rng.uniform(*solid)
    chars = rng.choices("#XE.", weights=(solid, spike, exit_, 1 - solid - spike - exit_), k=GRID_W * GRID_H)
    grid = [chars[r * GRID_W:(r + 1) * GRID_W] for r in range(GRID_H)]
    for _ in range(TRIES):
        c, r = rng.randrange(GRID_W - SIZE + 1), rng.randrange(GRID_H - SIZE + 1)
        if all(grid[r + dr][c + dc] == "." for dr in range(SIZE) for dc in range(SIZE)):
            grid[r][c] = "S"
            text = "\n".join("".join(row) for row in grid) + "\n"
            return text, parse_level(text)
    return None


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


def generate(count, level_path=None, solid=(0.02, 0.2), spike=0.03, exit_=0.01, grounded=0.5, seed=None):
    """level_path=None makes a new random map for every pair."""
    rng = random.Random(seed)
    if level_path is not None:
        fixed = (Path(level_path).read_text(), load_level(level_path))
    pairs = []
    while len(pairs) < count:
        made = fixed if level_path is not None else random_level(rng, solid, spike, exit_)
        if made is None:
            continue
        level_text, level = made
        if rng.random() < grounded:
            state = random_grounded_state(level, rng)
        else:
            state = random_air_state(level, rng)
        if state is None:
            if level_path is not None:
                raise ValueError(f"no valid start position found in {level_path}")
            continue  # this random map had no valid spot; make another
        action = (rng.choice((-1, 0, 1)), rng.choice((0, 1)))
        _, dead, new_state, finished = advance(state, action, level)
        pairs.append(make_pair(level_text, state, action, new_state, dead, finished))
    return pairs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-n", "--count", type=int, default=1000, help="number of pairs (default 1000)")
    parser.add_argument("--level", help="use this level file for every pair instead of random maps")
    parser.add_argument("--solid", type=float, nargs=2, default=(0.02, 0.2), metavar=("MIN", "MAX"),
                        help="range each map's wall chance is drawn from (default 0.02 0.2)")
    parser.add_argument("--spike", type=float, default=0.03, help="chance a random tile is a spike (default 0.03)")
    parser.add_argument("--exit", type=float, default=0.01, help="chance a random tile is an exit (default 0.01)")
    parser.add_argument("--grounded", type=float, default=0.5, help="fraction starting on a floor (default 0.5)")
    parser.add_argument("--seed", type=int, help="random seed, for repeatable output")
    args = parser.parse_args()
    if not 0 <= args.solid[0] <= args.solid[1] or args.solid[1] + args.spike + args.exit > 1:
        parser.error("need 0 <= --solid MIN <= MAX, and MAX + --spike + --exit at most 1")

    pairs = generate(args.count, args.level, args.solid, args.spike, args.exit, args.grounded, args.seed)
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    path = GENERATED_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(json.dumps(pairs, indent=2))
    print(f"Generated {len(pairs)} pairs to {path}")
