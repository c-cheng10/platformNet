
import argparse
import json
import random
import time

from map_gen import make_maps_np

from gamerecorder import make_pair
from generate_clumped_data import GENERATED_DIR, SIZE, random_air_state, random_grounded_state
from generate_grounded_spawn_data import FLAT, flat_spots, flat_width, tiles_to_text
from platformnet.level import parse_level, resolve_level
from platformnet.physics import SOLID
from platformnet.run import advance

GROUNDED = 1.0  # Share of pairs starting on a floor; the rest start in the air
JUMP = 0.75     # Share of pairs holding jump


def grounded_mapgen_levels(count, rng, flat=FLAT):

    levels = []
    while len(levels) < count:
        for tiles in make_maps_np(count - len(levels), seed=rng.randrange(2**32)):
            tiles = tiles.tolist()

            # Redraw maps with nowhere flat enough to spawn
            spots = flat_spots(tiles, flat)
            if not spots:
                continue

            text = tiles_to_text(tiles, rng.choice(spots))
            level = parse_level(text)
            assert flat_width(level) >= flat
            levels.append((text, level))
    return levels


def random_action(rng, jump):
    # Move left, stay or right; hold jump with probability `jump`
    move = rng.choice((-1, 0, 1))
    wants_jump = int(rng.random() < jump)
    return (move, wants_jump)


def generate(count, maps=100, grounded=GROUNDED, jump=JUMP, seed=None, flat=FLAT):

    rng = random.Random(seed)

    levels = grounded_mapgen_levels(maps, rng, flat)

    pairs = []

    while len(pairs) < count:

        # Every level is equally likely
        index = rng.randrange(len(levels))
        level = levels[index][1]

        if rng.random() < grounded:
            state = random_grounded_state(level, rng)
        else:
            state = random_air_state(level, rng)

        # No valid spot found this time (rare); draw again
        if state is None:
            continue

        assert not level.overlaps(state, SOLID), "start state inside a wall"

        action = random_action(rng, jump)
        _, dead, new_state, finished = advance(state, action, level)
        pairs.append(make_pair(index, state, action, new_state, dead, finished))

    level_texts = [text for text, _ in levels]
    return level_texts, pairs


def save(levels, pairs):
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    path = GENERATED_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}-mapgen-jump.json"
    path.write_text(json.dumps({"levels": levels, "pairs": pairs}, separators=(",", ":")))
    return path


def save_preview(name, seed, flat):
    # Write one map to levels/NAME.txt so it can be opened in level_editor.py
    path = resolve_level(name)
    text, _ = grounded_mapgen_levels(1, random.Random(seed), flat)[0]
    path.write_text(text)
    return path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pairs on map_gen.py maps only, starting on the ground and mostly jumping.")
    parser.add_argument("-n", "--count", type=int, default=250000, help="Number of pairs (default 250000)")
    parser.add_argument("--maps", type=int, default=100, help="map_gen levels per file (default 100)")
    parser.add_argument("--grounded", type=float, default=GROUNDED,
                        help=f"Fraction starting on a floor (default {GROUNDED})")
    parser.add_argument("--jump", type=float, default=JUMP, help=f"Fraction holding jump (default {JUMP})")
    parser.add_argument("--flat", type=int, default=FLAT,
                        help=f"Minimum tiles of flat floor under each spawn (default {FLAT})")
    parser.add_argument("--seed", type=int, help="Random seed for repeatable output")
    parser.add_argument("--preview", metavar="NAME",
                        help="Just write one map to levels/NAME.txt (open it in level_editor.py) and exit")
    args = parser.parse_args()

    # Preview mode skips the other checks
    if args.preview:
        return args

    if not 0 <= args.grounded <= 1:
        parser.error("--grounded must be between 0 and 1")
    if not 0 <= args.jump <= 1:
        parser.error("--jump must be between 0 and 1")
    if args.maps < 1:
        parser.error("--maps must be at least 1")
    if args.flat < SIZE:
        parser.error(f"--flat must be at least {SIZE} (the player's width)")
    return args


if __name__ == "__main__":
    args = parse_args()

    if args.preview:
        path = save_preview(args.preview, args.seed, args.flat)
        print(f"Wrote a preview map to {path}")
    else:
        levels, pairs = generate(args.count, args.maps, args.grounded, args.jump, args.seed, args.flat)
        path = save(levels, pairs)
        print(f"Generated {len(pairs)} pairs on {len(levels)} levels to {path}")
