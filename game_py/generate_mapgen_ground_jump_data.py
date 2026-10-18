"""Pairs on render_net/map_gen.py maps only, starting on the ground and mostly jumping.

    python generate_mapgen_ground_jump_data.py [-n COUNT] [--maps M] [--grounded FRAC] [--jump FRAC]
                                               [--flat TILES] [--seed SEED]
    python generate_mapgen_ground_jump_data.py --preview NAME [--seed SEED]   one map -> levels/NAME.txt

Every level is a map_gen map (no real levels). Each map's spawn S stands on a flat floor at least
--flat tiles wide (default 6), as in generate_grounded_spawn_data.py; maps with no such spot are
redrawn. --grounded of the pairs (default all) start standing exactly on a floor with vy = 0, the
rest anywhere in the air, never inside a wall. --jump of the pairs (default 75%) hold jump.

Output: generated/<timestamp>-mapgen-jump.json, minified: {"levels": [text, ...], "pairs": [...]},
with each pair's LEVEL an index into "levels".
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "render_net"))
from map_gen import make_maps_np  # noqa: E402

from gamerecorder import make_pair
from generate_clumped_data import GENERATED_DIR, SIZE, TILE_TO_CHAR, random_air_state, random_grounded_state
from generate_grounded_spawn_data import FLAT, flat_spots, flat_width
from platformnet.level import parse_level, resolve_level
from platformnet.physics import SOLID
from platformnet.run import advance

JUMP = 0.75  # share of pairs holding jump


def grounded_mapgen_levels(count, rng, flat=FLAT):
    """count map_gen maps, each with one spawn S standing on at least `flat` tiles of flat floor
    (maps with none are redrawn). Returns [(level text, Level), ...]."""
    levels = []
    while len(levels) < count:
        for tiles in make_maps_np(count - len(levels), seed=rng.randrange(2**32)):
            spots = flat_spots(tiles, flat)
            if not spots:
                continue
            grid = [[TILE_TO_CHAR[t] for t in row] for row in tiles.tolist()]
            c, r = rng.choice(spots)
            grid[r][c] = "S"
            text = "\n".join("".join(row) for row in grid) + "\n"
            level = parse_level(text)
            assert flat_width(level) >= flat
            levels.append((text, level))
    return levels


def generate(count, maps=100, grounded=1.0, jump=JUMP, seed=None, flat=FLAT):
    """Returns (level texts, pairs); each pair's LEVEL is an index into the level texts."""
    rng = random.Random(seed)
    levels = grounded_mapgen_levels(maps, rng, flat)
    pairs = []
    while len(pairs) < count:
        index = rng.randrange(len(levels))
        level = levels[index][1]
        if rng.random() < grounded:
            state = random_grounded_state(level, rng)
        else:
            state = random_air_state(level, rng)
        if state is None:  # no valid spot found this time (rare); draw again
            continue
        assert not level.overlaps(state, SOLID), "start state inside a wall"
        action = (rng.choice((-1, 0, 1)), int(rng.random() < jump))
        _, dead, new_state, finished = advance(state, action, level)
        pairs.append(make_pair(index, state, action, new_state, dead, finished))
    return [text for text, _ in levels], pairs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-n", "--count", type=int, default=250000, help="number of pairs (default 250000)")
    parser.add_argument("--maps", type=int, default=100, help="map_gen levels per file (default 100)")
    parser.add_argument("--grounded", type=float, default=1.0, help="fraction starting on a floor (default 1)")
    parser.add_argument("--jump", type=float, default=JUMP, help=f"fraction holding jump (default {JUMP})")
    parser.add_argument("--flat", type=int, default=FLAT,
                        help=f"minimum tiles of flat floor under each spawn (default {FLAT})")
    parser.add_argument("--seed", type=int, help="random seed, for repeatable output")
    parser.add_argument("--preview", metavar="NAME",
                        help="just write one map to levels/NAME.txt (open it in level_editor.py) and exit")
    args = parser.parse_args()
    if args.preview:
        path = resolve_level(args.preview)
        path.write_text(grounded_mapgen_levels(1, random.Random(args.seed), args.flat)[0][0])
        print(f"Wrote a preview map to {path}")
        sys.exit()
    for name in ("grounded", "jump"):
        if not 0 <= getattr(args, name) <= 1:
            parser.error(f"--{name} must be between 0 and 1")
    if args.maps < 1:
        parser.error("--maps must be at least 1")
    if args.flat < SIZE:
        parser.error(f"--flat must be at least {SIZE} (the player's width)")

    levels, pairs = generate(args.count, args.maps, args.grounded, args.jump, args.seed, args.flat)
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    path = GENERATED_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}-mapgen-jump.json"
    path.write_text(json.dumps({"levels": levels, "pairs": pairs}, separators=(",", ":")))
    print(f"Generated {len(pairs)} pairs on {len(levels)} levels to {path}")
