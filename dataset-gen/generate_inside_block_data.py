"""Pairs on clumped cave maps where the player starts inside a block and moves up one block.

    python generate_inside_block_data.py [-n COUNT] [--maps M] [--seed SEED]

Maps are clumped caves as in generate_clumped_data.py. Every pair starts at a random position
where the player overlaps a solid tile, with zero velocity and no input (no left, right or jump).
The next frame is the same, except y is one tile (TILE px) higher. isDead and isFinished are 0.

Output: generated/<timestamp>-inside-block.json, minified: {"levels": [text, ...], "pairs": [...]},
with each pair's LEVEL an index into "levels".
"""
import argparse
import json
import random
import time

from gamerecorder import make_pair
from generate_clumped_data import GENERATED_DIR, MAX_X, MAX_Y, TRIES, clumped_levels
from platformnet.physics import SOLID, TILE

NO_INPUT = (0, 0)


def random_inside_state(level, rng):
    """A still state overlapping a solid tile, low enough that moving up a tile stays on the map."""
    for _ in range(TRIES):
        state = (rng.uniform(0, MAX_X), rng.uniform(TILE, MAX_Y), 0.0, 0.0)
        if level.overlaps(state, SOLID):
            return state
    return None


def generate(count, maps=100, seed=None):
    """Returns (level texts, pairs); each pair's LEVEL is an index into the level texts."""
    rng = random.Random(seed)
    levels = clumped_levels(maps, rng)
    pairs = []
    while len(pairs) < count:
        index = rng.randrange(len(levels))
        state = random_inside_state(levels[index][1], rng)
        if state is None:  # no solid tile found this time (rare); draw again
            continue
        x, y, _, _ = state
        pairs.append(make_pair(index, state, NO_INPUT, (x, y - TILE, 0.0, 0.0), False, False))
    return [text for text, _ in levels], pairs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-n", "--count", type=int, default=250000, help="number of pairs (default 250000)")
    parser.add_argument("--maps", type=int, default=100, help="levels per file (default 100)")
    parser.add_argument("--seed", type=int, help="random seed, for repeatable output")
    args = parser.parse_args()
    if args.maps < 1:
        parser.error("--maps must be at least 1")

    levels, pairs = generate(args.count, args.maps, args.seed)
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    path = GENERATED_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}-inside-block.json"
    path.write_text(json.dumps({"levels": levels, "pairs": pairs}, separators=(",", ":")))
    print(f"Generated {len(pairs)} pairs on {len(levels)} levels to {path}")
