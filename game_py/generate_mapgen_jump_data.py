"""Pairs on a fixed set of real levels, half starting on the ground and half in the air, mostly jumping.

    python generate_mapgen_jump_data.py [-n COUNT] [--levels NAME ...] [--grounded FRAC] [--jump FRAC]
                                        [--seed SEED]

Every pair uses one of --levels (default level1, level2, level3, traininglevel_181635,
traininglevel_181707, traininglevel3, trainlevel1), picked evenly. --grounded of the pairs (default
50%) start standing exactly on a floor with vy = 0, the rest anywhere in the air; never inside a
wall, and never touching spikes or the exit. --jump of the pairs (default 75%) hold jump.

Output: generated/<timestamp>-levels-jump.json, minified: {"levels": [text, ...], "pairs": [...]},
with each pair's LEVEL an index into "levels".
"""
import argparse
import json
import random
import time

from gamerecorder import make_pair
from generate_clumped_data import GENERATED_DIR, random_air_state, random_grounded_state
from platformnet.level import load_level, resolve_level
from platformnet.physics import SOLID
from platformnet.run import advance

DEFAULT_LEVELS = ("level1", "level2", "level3", "traininglevel_181635", "traininglevel_181707",
                  "traininglevel3", "trainlevel1")
GROUNDED = 0.5  # share of pairs starting on a floor; the rest start in the air
JUMP = 0.75  # share of pairs holding jump


def generate(count, level_names=DEFAULT_LEVELS, grounded=GROUNDED, jump=JUMP, seed=None):
    """Returns (level texts, pairs); each pair's LEVEL is an index into the level texts."""
    rng = random.Random(seed)
    paths = [resolve_level(name) for name in level_names]
    levels = [(p.read_text(), load_level(p)) for p in paths]
    pairs = []
    misses = [0] * len(levels)
    while len(pairs) < count:
        index = rng.randrange(len(levels))
        level = levels[index][1]
        if rng.random() < grounded:
            state = random_grounded_state(level, rng)
        else:
            state = random_air_state(level, rng)
        if state is None:  # no valid spot found this time (rare); draw again
            misses[index] += 1
            if misses[index] > 100:
                raise ValueError(f"no valid start position found in {paths[index]}")
            continue
        assert not level.overlaps(state, SOLID), "start state inside a wall"
        action = (rng.choice((-1, 0, 1)), int(rng.random() < jump))
        _, dead, new_state, finished = advance(state, action, level)
        pairs.append(make_pair(index, state, action, new_state, dead, finished))
    return [text for text, _ in levels], pairs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-n", "--count", type=int, default=250000, help="number of pairs (default 250000)")
    parser.add_argument("--levels", nargs="+", default=list(DEFAULT_LEVELS),
                        help="levels to use, picked evenly (default: " + " ".join(DEFAULT_LEVELS) + ")")
    parser.add_argument("--grounded", type=float, default=GROUNDED,
                        help=f"fraction starting on a floor, the rest in the air (default {GROUNDED})")
    parser.add_argument("--jump", type=float, default=JUMP, help=f"fraction holding jump (default {JUMP})")
    parser.add_argument("--seed", type=int, help="random seed, for repeatable output")
    args = parser.parse_args()
    for name in ("grounded", "jump"):
        if not 0 <= getattr(args, name) <= 1:
            parser.error(f"--{name} must be between 0 and 1")

    levels, pairs = generate(args.count, args.levels, args.grounded, args.jump, args.seed)
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    path = GENERATED_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}-levels-jump.json"
    path.write_text(json.dumps({"levels": levels, "pairs": pairs}, separators=(",", ":")))
    print(f"Generated {len(pairs)} pairs on {len(levels)} levels to {path}")
