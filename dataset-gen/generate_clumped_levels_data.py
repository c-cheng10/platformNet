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

    rng = random.Random(seed)
    
    paths = [resolve_level(name) for name in level_names] if level_share > 0 else []
    fixed = [(p.read_text(), load_level(p)) for p in paths]
    
    # Calculate how many random clumped maps to generate
    n_random = maps - len(fixed) if level_share < 1 else 0
    levels = fixed + clumped_levels(n_random, rng)
    
    pairs = []
    misses = [0] * len(fixed)

    # Generate pairs until we hit the target count
    while len(pairs) < count:
        use_fixed = bool(fixed) and (not n_random or rng.random() < level_share)
        index = rng.randrange(len(fixed)) if use_fixed else len(fixed) + rng.randrange(n_random)
        level = levels[index][1]

        if rng.random() < grounded:
            state = random_grounded_state(level, rng)
        else:
            state = random_air_state(level, rng)

        if state is None:
            if use_fixed:
                misses[index] += 1
                if misses[index] > 100:
                    raise ValueError(f"No valid start position found in {paths[index]}")
            continue

        # Simulate a random action and record the resulting pair
        action = (rng.choice((-1, 0, 1)), rng.choice((0, 1)))
        _, dead, new_state, finished = advance(state, action, level)
        pairs.append(make_pair(index, state, action, new_state, dead, finished))

    level_texts = [text for text, _ in levels]
    return level_texts, pairs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-n", "--count", type=int, default=250000, help="Number of pairs (default 250000)")
    parser.add_argument("--levels", nargs="+", default=list(DEFAULT_LEVELS),
                        help="Real levels mixed in, picked evenly")
    parser.add_argument("--level-share", type=float, default=0.4,
                        help="Fraction of pairs on real levels; the rest use clumped maps (default 0.4)")
    parser.add_argument("--maps", type=int, default=100, help="Levels per file, including real levels (default 100)")
    parser.add_argument("--grounded", type=float, default=0.5, help="Fraction starting on a floor (default 0.5)")
    parser.add_argument("--seed", type=int, help="Random seed for repeatable output")
    
    args = parser.parse_args()

    # Validate command-line arguments
    if not 0 <= args.level_share <= 1:
        parser.error("--level-share must be between 0 and 1")
    if 0 < args.level_share < 1 and args.maps <= len(args.levels):
        parser.error(f"--maps must be more than {len(args.levels)} when mixing in clumped maps")
    if args.level_share == 0 and args.maps < 1:
        parser.error("--maps must be at least 1")

    # Generate data and save to a minified JSON file
    levels, pairs = generate(args.count, args.levels, args.level_share, args.maps, args.grounded, args.seed)
    
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    path = GENERATED_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}-clumped-levels.json"
    path.write_text(json.dumps({"levels": levels, "pairs": pairs}, separators=(",", ":")))
    
    print(f"Generated {len(pairs)} pairs on {len(levels)} levels to {path}")