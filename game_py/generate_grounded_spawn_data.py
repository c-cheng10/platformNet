"""Like generate_clumped_data.py, but spawns always stand on flat ground and never sit in a wall.

    python generate_grounded_spawn_data.py [-n COUNT] [--levels FILE ...] [--level-share FRAC] [--maps M]
                                           [--grounded FRAC] [--flat TILES] [--seed SEED]

Each clumped map's spawn S is picked from clear SIZE x SIZE spots (no solid, spike or exit tiles)
standing on a flat floor at least --flat tiles wide (default 6): a run of solid tiles with clear
space above, no steps, spikes or drops. Maps with no such spot are redrawn. The real levels
(--levels, default level1, level2, level3) are checked and rejected if their S is in a wall or not
on --flat tiles of flat floor. Every sampled start state is also checked not to overlap a wall.

A share of pairs (--level-share) use the real levels, split evenly between them; each file holds
--maps levels in total (default 100, real levels included). Maps and sampling otherwise match
generate_clumped_data.py.

Output: generated/<timestamp>-clumped-grounded.json
"""
import argparse
import json
import random
import time

import numpy as np

from gamerecorder import make_pair
from generate_clumped_data import (GENERATED_DIR, SIZE, TILE_TO_CHAR, is_reachable, make_clumped_map,
                                   random_air_state, random_grounded_state)
from platformnet.level import load_level, parse_level, resolve_level
from platformnet.physics import EMPTY, GRID_H, GRID_W, SOLID, TILE
from platformnet.run import advance

DEFAULT_LEVELS = ("level1", "level2", "level3")
FLAT = 6  # minimum width, in tiles, of the flat floor a spawn stands on


def flat_width(level):
    """Width in tiles of the flat floor under the spawn: the run of columns around it that are solid
    on the row below the player and empty for the player's height above (no steps, spikes or drops).
    0 if the spawn overlaps a wall, spike or exit, or isn't fully on solid ground."""
    if not is_reachable(level.spawn, level):
        return 0
    c0, r0 = int(level.spawn[0] // TILE), int(level.spawn[1] // TILE)

    def standable(c):
        return (level.tile(c, r0 + SIZE) == SOLID
                and all(level.tile(c, r) == EMPTY for r in range(r0, r0 + SIZE)))

    if not all(standable(c) for c in range(c0, c0 + SIZE)):
        return 0
    lo, hi = c0, c0 + SIZE - 1
    while standable(lo - 1):
        lo -= 1
    while standable(hi + 1):
        hi += 1
    return hi - lo + 1


def flat_spots(tiles, flat):
    """[(c, r), ...] top-left tiles where the player stands on a run of at least `flat` standable columns."""
    empty = tiles == EMPTY
    solid = tiles == SOLID
    spots = []
    for r in range(GRID_H - SIZE):
        standable = empty[r:r + SIZE].all(axis=0) & solid[r + SIZE]
        c = 0
        while c < GRID_W:  # walk the runs of standable columns
            if not standable[c]:
                c += 1
                continue
            start = c
            while c < GRID_W and standable[c]:
                c += 1
            if c - start >= flat:
                spots += [(x, r) for x in range(start, c - SIZE + 1)]
    return spots


def grounded_spawn_levels(count, rng, flat=FLAT):
    """count clumped maps, each with one spawn S standing on at least `flat` tiles of flat floor
    (maps with none are redrawn). Returns [(level text, Level), ...]."""
    np_rng = np.random.default_rng(rng.randrange(2**32))
    levels = []
    while len(levels) < count:
        tiles = make_clumped_map(np_rng)
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


def load_fixed_levels(names, flat=FLAT):
    """[(level text, Level), ...] for the real levels; raises if any spawn is in a wall or isn't on
    at least `flat` tiles of flat floor."""
    fixed = []
    for name in names:
        path = resolve_level(name)
        level = load_level(path)
        if not is_reachable(level.spawn, level):
            raise ValueError(f"spawn in {path} overlaps a wall, spike or exit")
        if flat_width(level) < flat:
            raise ValueError(f"spawn in {path} is on {flat_width(level)} tiles of flat floor, need {flat}")
        fixed.append((path.read_text(), level))
    return fixed


def generate(count, level_paths=DEFAULT_LEVELS, level_share=0.4, maps=100, grounded=0.5, seed=None,
             flat=FLAT):
    """Each pair uses one of level_paths (picked evenly) with chance level_share, otherwise one of
    the clumped maps; `maps` levels in total, level_paths included. Returns (level texts, pairs);
    each pair's LEVEL is an index into the level texts."""
    rng = random.Random(seed)
    fixed = load_fixed_levels(level_paths, flat) if level_share > 0 else []
    n_random = maps - len(fixed) if level_share < 1 else 0
    levels = fixed + grounded_spawn_levels(n_random, rng, flat)
    pairs = []
    misses = [0] * len(levels)
    while len(pairs) < count:
        use_fixed = bool(fixed) and (not n_random or rng.random() < level_share)
        index = rng.randrange(len(fixed)) if use_fixed else len(fixed) + rng.randrange(n_random)
        level = levels[index][1]
        if rng.random() < grounded:
            state = random_grounded_state(level, rng)
        else:
            state = random_air_state(level, rng)
        if state is None:  # no valid spot found this time (rare); draw again
            misses[index] += 1
            if index < len(fixed) and misses[index] > 100:
                raise ValueError(f"no valid start position found in {resolve_level(level_paths[index])}")
            continue
        assert not level.overlaps(state, SOLID), "start state inside a wall"
        action = (rng.choice((-1, 0, 1)), rng.choice((0, 1)))
        _, dead, new_state, finished = advance(state, action, level)
        pairs.append(make_pair(index, state, action, new_state, dead, finished))
    return [text for text, _ in levels], pairs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-n", "--count", type=int, default=250000, help="number of pairs (default 250000)")
    parser.add_argument("--levels", nargs="+", default=list(DEFAULT_LEVELS),
                        help="real levels mixed in (default level1 level2 level3)")
    parser.add_argument("--level-share", type=float, default=0.4,
                        help="fraction of pairs on --levels; the rest get clumped maps (default 0.4; 1 = all, 0 = none)")
    parser.add_argument("--maps", type=int, default=100, help="levels per file, --levels included (default 100)")
    parser.add_argument("--grounded", type=float, default=0.5, help="fraction starting on a floor (default 0.5)")
    parser.add_argument("--flat", type=int, default=FLAT,
                        help=f"minimum tiles of flat floor under each spawn (default {FLAT})")
    parser.add_argument("--seed", type=int, help="random seed, for repeatable output")
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
