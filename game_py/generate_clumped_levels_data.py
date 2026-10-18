"""Like generate_clumped_data.py, but the real levels mixed in are a fixed set instead of one --level.

    python generate_clumped_levels_data.py [-n COUNT] [--levels NAME ...] [--level-share FRAC] [--maps M]
                                           [--grounded FRAC] [--seed SEED]

A share of pairs (--level-share, default 0.4) use one of --levels (default level1, level2, level3,
traininglevel_181635, traininglevel_181707, traininglevel3, trainlevel1), picked evenly; the rest
use clumped cave maps as in generate_clumped_data.py. Each file holds --maps levels in total
(default 100, --levels included), frame 1 is always a reachable state, and --grounded of the
samples stand exactly on a floor.

Output: generated/<timestamp>-clumped-levels.json, minified: {"levels": [text, ...], "pairs": [...]},
with each pair's LEVEL an index into "levels".
"""
import argparse
import json
import random
import time

from gamerecorder import make_pair
from generate_clumped_data import GENERATED_DIR, clumped_levels, random_air_state, random_grounded_state
from generate_mapgen_jump_data import DEFAULT_LEVELS
from platformnet.level import load_level, resolve_level
from platformnet.run import advance


def generate(count, level_names=DEFAULT_LEVELS, level_share=0.4, maps=100, grounded=0.5, seed=None):
    """Each pair uses one of level_names (picked evenly) with chance level_share, otherwise one of
    the clumped maps; `maps` levels in total, level_names included. Returns (level texts, pairs);
    each pair's LEVEL is an index into the level texts."""
    rng = random.Random(seed)
    paths = [resolve_level(name) for name in level_names] if level_share > 0 else []
    fixed = [(p.read_text(), load_level(p)) for p in paths]
    n_random = maps - len(fixed) if level_share < 1 else 0
    levels = fixed + clumped_levels(n_random, rng)
    pairs = []
    misses = [0] * len(fixed)
    while len(pairs) < count:
        use_fixed = bool(fixed) and (not n_random or rng.random() < level_share)
        index = rng.randrange(len(fixed)) if use_fixed else len(fixed) + rng.randrange(n_random)
        level = levels[index][1]
        if rng.random() < grounded:
            state = random_grounded_state(level, rng)
        else:
            state = random_air_state(level, rng)
        if state is None:  # no valid spot found this time (rare); draw again
            if use_fixed:
                misses[index] += 1
                if misses[index] > 100:
                    raise ValueError(f"no valid start position found in {paths[index]}")
            continue
        action = (rng.choice((-1, 0, 1)), rng.choice((0, 1)))
        _, dead, new_state, finished = advance(state, action, level)
        pairs.append(make_pair(index, state, action, new_state, dead, finished))
    return [text for text, _ in levels], pairs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-n", "--count", type=int, default=250000, help="number of pairs (default 250000)")
    parser.add_argument("--levels", nargs="+", default=list(DEFAULT_LEVELS),
                        help="real levels mixed in, picked evenly (default: " + " ".join(DEFAULT_LEVELS) + ")")
    parser.add_argument("--level-share", type=float, default=0.4,
                        help="fraction of pairs on --levels; the rest get clumped maps (default 0.4; 1 = all, 0 = none)")
    parser.add_argument("--maps", type=int, default=100, help="levels per file, --levels included (default 100)")
    parser.add_argument("--grounded", type=float, default=0.5, help="fraction starting on a floor (default 0.5)")
    parser.add_argument("--seed", type=int, help="random seed, for repeatable output")
    args = parser.parse_args()
    if not 0 <= args.level_share <= 1:
        parser.error("--level-share must be between 0 and 1")
    if 0 < args.level_share < 1 and args.maps <= len(args.levels):
        parser.error(f"--maps must be more than {len(args.levels)} (the --levels) when mixing in clumped maps")
    if args.level_share == 0 and args.maps < 1:
        parser.error("--maps must be at least 1")

    levels, pairs = generate(args.count, args.levels, args.level_share, args.maps, args.grounded, args.seed)
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    path = GENERATED_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}-clumped-levels.json"
    path.write_text(json.dumps({"levels": levels, "pairs": pairs}, separators=(",", ":")))
    print(f"Generated {len(pairs)} pairs on {len(levels)} levels to {path}")
