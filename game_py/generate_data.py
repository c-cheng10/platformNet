"""Generate random input -> output pairs without playing, all into one file.

    python generate_data.py [-n COUNT] [--level FILE] [--grounded FRAC] [--seed SEED]

Each pair: a random frame 1 (position, velocity, keys) and frame 2 computed by the real physics,
in the same format as gamerecorder.py (frame 2 is the spawn point after a death; isDead and
isFinished belong to frame 1). Frame 1 never starts inside a wall. A fraction of samples (--grounded)
stand exactly on a floor with vy = 0, since random positions almost never land on the ground and
jumping / ground friction would otherwise never appear.

Output: generated/<timestamp>.json
"""
import argparse
import json
import random
import time
from pathlib import Path

from gamerecorder import make_pair
from platformnet.level import LEVEL_PATH, load_level
from platformnet.physics import GRID_H, GRID_W, JUMP_VEL, MAX_FALL, MAX_RUN, PLAYER, SOLID, TILE
from platformnet.run import advance

GENERATED_DIR = Path(__file__).resolve().parent / "generated"
MAX_X = GRID_W * TILE - PLAYER
MAX_Y = GRID_H * TILE - PLAYER


def random_air_state(level, rng):
    while True:
        state = (rng.uniform(0, MAX_X), rng.uniform(0, MAX_Y),
                 rng.uniform(-MAX_RUN, MAX_RUN), rng.uniform(JUMP_VEL, MAX_FALL))
        if not level.overlaps(state, SOLID):
            return state


def random_grounded_state(level, rng):
    while True:
        x = rng.uniform(0, MAX_X)
        floor_row = rng.randrange(PLAYER // TILE, GRID_H)
        state = (x, float(floor_row * TILE - PLAYER), rng.uniform(-MAX_RUN, MAX_RUN), 0.0)
        under = range(int(x // TILE), int(-(-(x + PLAYER) // TILE)))
        if any(level.tile(c, floor_row) == SOLID for c in under) and not level.overlaps(state, SOLID):
            return state


def generate(count, level_path=LEVEL_PATH, grounded=0.5, seed=None):
    rng = random.Random(seed)
    level_text = Path(level_path).read_text()
    level = load_level(level_path)
    pairs = []
    for _ in range(count):
        if rng.random() < grounded:
            state = random_grounded_state(level, rng)
        else:
            state = random_air_state(level, rng)
        action = (rng.choice((-1, 0, 1)), rng.choice((0, 1)))
        _, dead, new_state, finished = advance(state, action, level)
        pairs.append(make_pair(level_text, state, action, new_state, dead, finished))
    return pairs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-n", "--count", type=int, default=1000, help="number of pairs (default 1000)")
    parser.add_argument("--level", default=LEVEL_PATH, help="level file (default levels/level1.txt)")
    parser.add_argument("--grounded", type=float, default=0.5, help="fraction starting on a floor (default 0.5)")
    parser.add_argument("--seed", type=int, help="random seed, for repeatable output")
    args = parser.parse_args()

    pairs = generate(args.count, args.level, args.grounded, args.seed)
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    path = GENERATED_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(json.dumps(pairs, indent=2))
    print(f"Generated {len(pairs)} pairs to {path}")
