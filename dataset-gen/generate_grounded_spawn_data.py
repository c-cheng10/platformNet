import argparse
import json
import random
import time

from gamerecorder import make_pair
from generate_clumped_data import (GENERATED_DIR, SIZE, TILE_TO_CHAR, is_reachable, make_clumped_map,
                                   random_air_state, random_grounded_state)
from platformnet.level import load_level, parse_level, resolve_level
from platformnet.physics import EMPTY, GRID_H, GRID_W, SOLID, TILE
from platformnet.run import advance

DEFAULT_LEVELS = ("level1", "level2", "level3")
FLAT = 6
MAX_MISSES = 100


def flat_width(level):

    if not is_reachable(level.spawn, level):
        return 0

    spawn_c = int(level.spawn[0] // TILE)
    spawn_r = int(level.spawn[1] // TILE)

    def standable(c):
        if level.tile(c, spawn_r + SIZE) != SOLID:
            return False
        for r in range(spawn_r, spawn_r + SIZE):
            if level.tile(c, r) != EMPTY:
                return False
        return True

    for c in range(spawn_c, spawn_c + SIZE):
        if not standable(c):
            return 0

    left = spawn_c
    right = spawn_c + SIZE - 1
    while standable(left - 1):
        left -= 1
    while standable(right + 1):
        right += 1
    return right - left + 1


def is_standable(tiles, c, r):
    if tiles[r + SIZE][c] != SOLID:
        return False
    for pr in range(r, r + SIZE):
        if tiles[pr][c] != EMPTY:
            return False
    return True


def flat_spots(tiles, flat):
    spots = []

    for r in range(GRID_H - SIZE):
        c = 0
        while c < GRID_W:
            if not is_standable(tiles, c, r):
                c += 1
                continue

            start = c
            while c < GRID_W and is_standable(tiles, c, r):
                c += 1
            run_length = c - start

            if run_length >= flat:
                for x in range(start, c - SIZE + 1):
                    spots.append((x, r))
    return spots


def tiles_to_text(tiles, start):
    grid_chars = []
    for row in tiles:
        char_row = [TILE_TO_CHAR[t] for t in row]
        grid_chars.append(char_row)

    spawn_c, spawn_r = start
    grid_chars[spawn_r][spawn_c] = "S"
    return "\n".join("".join(row) for row in grid_chars) + "\n"


def grounded_spawn_levels(count, rng, flat=FLAT):

    levels = []
    while len(levels) < count:
        tiles = make_clumped_map(rng)

        spots = flat_spots(tiles, flat)
        if not spots:
            continue

        text = tiles_to_text(tiles, rng.choice(spots))
        level = parse_level(text)
        assert flat_width(level) >= flat
        levels.append((text, level))
    return levels


def load_fixed_levels(names, flat=FLAT):

    paths = []
    levels = []
    for name in names:
        path = resolve_level(name)
        level = load_level(path)

        if not is_reachable(level.spawn, level):
            raise ValueError(f"spawn in {path} overlaps a wall, spike or exit")
        if flat_width(level) < flat:
            raise ValueError(f"spawn in {path} is on {flat_width(level)} tiles of flat floor, need {flat}")

        paths.append(path)
        levels.append((path.read_text(), level))
    return paths, levels


def generate(count, level_paths=DEFAULT_LEVELS, level_share=0.4, maps=100, grounded=0.5, seed=None,
             flat=FLAT):

    rng = random.Random(seed)

    paths = []
    fixed = []
    if level_share > 0:
        paths, fixed = load_fixed_levels(level_paths, flat)

    n_random = maps - len(fixed) if level_share < 1 else 0
    levels = fixed + grounded_spawn_levels(n_random, rng, flat)

    pairs = []
    misses = [0] * len(levels)

    while len(pairs) < count:

        if not fixed:
            use_fixed = False
        elif n_random == 0:
            use_fixed = True
        else:
            use_fixed = rng.random() < level_share

        if use_fixed:
            index = rng.randrange(len(fixed))
        else:
            index = len(fixed) + rng.randrange(n_random)
        level = levels[index][1]

        if rng.random() < grounded:
            state = random_grounded_state(level, rng)
        else:
            state = random_air_state(level, rng)

        if state is None:
            misses[index] += 1
            if use_fixed and misses[index] > MAX_MISSES:
                raise ValueError(f"no valid start position found in {paths[index]}")
            continue

        assert not level.overlaps(state, SOLID), "start state inside a wall"

        action = (rng.choice((-1, 0, 1)), rng.choice((0, 1)))
        _, dead, new_state, finished = advance(state, action, level)
        pairs.append(make_pair(index, state, action, new_state, dead, finished))

    return [text for text, _ in levels], pairs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Like generate_clumped_data.py, but spawns always stand on flat ground and never sit in a wall.")
    parser.add_argument("-n", "--count", type=int, default=250000, help="Number of pairs (default 250000)")
    parser.add_argument("--levels", nargs="+", default=list(DEFAULT_LEVELS),
                        help="Real levels mixed in (default level1 level2 level3)")
    parser.add_argument("--level-share", type=float, default=0.4,
                        help="Fraction of pairs on --levels; the rest get clumped maps (default 0.4; 1 = all, 0 = none)")
    parser.add_argument("--maps", type=int, default=100, help="Levels per file, --levels included (default 100)")
    parser.add_argument("--grounded", type=float, default=0.5, help="Fraction starting on a floor (default 0.5)")
    parser.add_argument("--flat", type=int, default=FLAT,
                        help=f"Minimum tiles of flat floor under each spawn (default {FLAT})")
    parser.add_argument("--seed", type=int, help="Random seed for repeatable output")
    args = parser.parse_args()

    if args.flat < SIZE:
        parser.error(f"--flat must be at least {SIZE} (the player's width)")
    if not 0 <= args.level_share <= 1:
        parser.error("--level-share must be between 0 and 1")
    if 0 < args.level_share < 1 and args.maps <= len(args.levels):
        parser.error("--maps must be more than the number of --levels when mixing in clumped maps")
    if args.level_share == 0 and args.maps < 1:
        parser.error("--maps must be at least 1")

    levels, pairs = generate(args.count, args.levels, args.level_share, args.maps, args.grounded, args.seed,
                             args.flat)

    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    path = GENERATED_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}-clumped-grounded.json"
    path.write_text(json.dumps({"levels": levels, "pairs": pairs}, separators=(",", ":")))

    print(f"Generated {len(pairs)} pairs on {len(levels)} levels to {path}")
