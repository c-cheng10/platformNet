
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
GROUNDED = 0.5
JUMP = 0.75
MAX_MISSES = 100


def load_levels(level_names):

    paths = []
    levels = []
    for name in level_names:
        path = resolve_level(name)
        text = path.read_text()
        level = load_level(path)

        paths.append(path)
        levels.append((text, level))
    return paths, levels


def random_action(rng, jump):
    move = rng.choice((-1, 0, 1))
    wants_jump = int(rng.random() < jump)
    return (move, wants_jump)


def generate(count, level_names=DEFAULT_LEVELS, grounded=GROUNDED, jump=JUMP, seed=None):

    rng = random.Random(seed)

    paths, levels = load_levels(level_names)

    pairs = []
    misses = [0] * len(levels)

    while len(pairs) < count:

        index = rng.randrange(len(levels))
        level = levels[index][1]

        if rng.random() < grounded:
            state = random_grounded_state(level, rng)
        else:
            state = random_air_state(level, rng)

        if state is None:
            misses[index] += 1
            if misses[index] > MAX_MISSES:
                raise ValueError(f"no valid start position found in {paths[index]}")
            continue

        assert not level.overlaps(state, SOLID), "start state inside a wall"

        action = random_action(rng, jump)
        _, dead, new_state, finished = advance(state, action, level)
        pairs.append(make_pair(index, state, action, new_state, dead, finished))

    level_texts = [text for text, _ in levels]
    return level_texts, pairs


def save(levels, pairs):
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    path = GENERATED_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}-levels-jump.json"
    path.write_text(json.dumps({"levels": levels, "pairs": pairs}, separators=(",", ":")))
    return path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pairs on a fixed set of real levels, half starting on the ground and half in the air, mostly jumping.")
    parser.add_argument("-n", "--count", type=int, default=250000, help="Number of pairs (default 250000)")
    parser.add_argument("--levels", nargs="+", default=list(DEFAULT_LEVELS),
                        help="Levels to use, picked evenly (default: " + " ".join(DEFAULT_LEVELS) + ")")
    parser.add_argument("--grounded", type=float, default=GROUNDED,
                        help=f"Fraction starting on a floor, the rest in the air (default {GROUNDED})")
    parser.add_argument("--jump", type=float, default=JUMP, help=f"Fraction holding jump (default {JUMP})")
    parser.add_argument("--seed", type=int, help="Random seed for repeatable output")
    args = parser.parse_args()

    if not 0 <= args.grounded <= 1:
        parser.error("--grounded must be between 0 and 1")
    if not 0 <= args.jump <= 1:
        parser.error("--jump must be between 0 and 1")
    return args


if __name__ == "__main__":
    args = parse_args()
    levels, pairs = generate(args.count, args.levels, args.grounded, args.jump, args.seed)
    path = save(levels, pairs)
    print(f"Generated {len(pairs)} pairs on {len(levels)} levels to {path}")
