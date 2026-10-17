"""Generate random input -> output pairs without playing, all into one file.

    python generate_data.py [-n COUNT] [--level FILE] [--level-share FRAC] [--solid MIN MAX]
                            [--spike P] [--exit P] [--cluster C] [--grounded FRAC] [--seed SEED]

A share of pairs (--level-share, default 40%) use a real level (--level, default levels/level1.txt).
Every other pair gets its own random map of walls, spikes (--spike), exits (--exit) and air, plus
one random spawn point S. Tiles clump together: filling the grid row by row, each tile copies an
already-placed neighbour (left, up-left, up, up-right) with chance --cluster, otherwise it is drawn
fresh. That makes wall chunks and air pockets instead of noise, while keeping the overall mix. Each
map picks its own wall chance between --solid MIN and MAX (default 2%-20%), so some maps are sparse
and some crowded, but air always far outnumbers walls. The map's text is the pair's LEVEL.

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
LEVEL1_PATH = Path(__file__).resolve().parent / "levels" / "level1.txt"
MAX_X = GRID_W * TILE - PLAYER
MAX_Y = GRID_H * TILE - PLAYER
SIZE = PLAYER // TILE  # player is SIZE x SIZE tiles
TRIES = 1000  # placement attempts before giving up on a map


def is_reachable(state, level):
    """Alive and still playing: not in a wall, not touching spikes (you'd have died) or the exit (you'd be done)."""
    return not any(level.overlaps(state, t) for t in (SOLID, SPIKE, EXIT))


def random_level(rng, solid=(0.02, 0.2), spike=0.03, exit_=0.01, cluster=0.85):
    """A map of clumped random tiles plus one spawn S on a clear SIZE x SIZE spot.
    The wall chance is drawn uniformly from the solid (min, max) range, once per map.
    Each tile copies a random already-placed neighbour with chance `cluster`, else takes a fresh
    draw; copying keeps the overall mix at the given chances, it just groups like tiles together.
    Returns (level text, Level), or None if no clear spot was found."""
    solid = rng.uniform(*solid)
    fresh = rng.choices("#XE.", weights=(solid, spike, exit_, 1 - solid - spike - exit_), k=GRID_W * GRID_H)
    grid = [["."] * GRID_W for _ in range(GRID_H)]
    for r in range(GRID_H):
        for c in range(GRID_W):
            neighbours = [grid[r][c - 1]] if c else []
            if r:
                neighbours += grid[r - 1][max(c - 1, 0):c + 2]
            if neighbours and rng.random() < cluster:
                grid[r][c] = rng.choice(neighbours)
            else:
                grid[r][c] = fresh[r * GRID_W + c]
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


def generate(count, level_path=LEVEL1_PATH, level_share=0.4, solid=(0.02, 0.2), spike=0.03, exit_=0.01,
             cluster=0.85, grounded=0.5, seed=None):
    """Each pair uses level_path with chance level_share, otherwise a new random map."""
    rng = random.Random(seed)
    fixed = (Path(level_path).read_text(), load_level(level_path)) if level_share > 0 else None
    pairs = []
    while len(pairs) < count:
        use_fixed = rng.random() < level_share
        made = fixed if use_fixed else random_level(rng, solid, spike, exit_, cluster)
        if made is None:
            continue
        level_text, level = made
        if rng.random() < grounded:
            state = random_grounded_state(level, rng)
        else:
            state = random_air_state(level, rng)
        if state is None:
            if use_fixed:
                raise ValueError(f"no valid start position found in {level_path}")
            continue  # this random map had no valid spot; make another
        action = (rng.choice((-1, 0, 1)), rng.choice((0, 1)))
        _, dead, new_state, finished = advance(state, action, level)
        pairs.append(make_pair(level_text, state, action, new_state, dead, finished))
    return pairs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-n", "--count", type=int, default=5000, help="number of pairs (default 100000)")
    parser.add_argument("--level", default=LEVEL1_PATH, help="real level mixed in (default levels/level1.txt)")
    parser.add_argument("--level-share", type=float, default=0.4,
                        help="fraction of pairs on --level; the rest get random maps (default 0.4; 1 = all, 0 = none)")
    parser.add_argument("--solid", type=float, nargs=2, default=(0.02, 0.2), metavar=("MIN", "MAX"),
                        help="range each map's wall chance is drawn from (default 0.02 0.2)")
    parser.add_argument("--spike", type=float, default=0.03, help="chance a random tile is a spike (default 0.03)")
    parser.add_argument("--exit", type=float, default=0.01, help="chance a random tile is an exit (default 0.01)")
    parser.add_argument("--cluster", type=float, default=0.85,
                        help="chance a tile copies a neighbour; 0 = pure noise, near 1 = big blobs (default 0.85)")
    parser.add_argument("--grounded", type=float, default=0.5, help="fraction starting on a floor (default 0.5)")
    parser.add_argument("--seed", type=int, help="random seed, for repeatable output")
    args = parser.parse_args()
    if not 0 <= args.solid[0] <= args.solid[1] or args.solid[1] + args.spike + args.exit > 1:
        parser.error("need 0 <= --solid MIN <= MAX, and MAX + --spike + --exit at most 1")
    if not 0 <= args.level_share <= 1:
        parser.error("--level-share must be between 0 and 1")

    pairs = generate(args.count, args.level, args.level_share, args.solid, args.spike, args.exit, args.cluster,
                     args.grounded, args.seed)
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    path = GENERATED_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(json.dumps(pairs, indent=2))
    print(f"Generated {len(pairs)} pairs to {path}")
