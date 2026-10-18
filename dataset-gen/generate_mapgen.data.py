
import argparse
import json
import random
import time
from pathlib import Path

from map_gen import make_maps_np
from gamerecorder import make_pair
from platformnet.level import LEVEL_PATH, load_level, parse_level, resolve_level
from platformnet.physics import EMPTY, EXIT, GRID_H, GRID_W, JUMP_VEL, MAX_FALL, MAX_RUN, PLAYER, SOLID, SPIKE, TILE
from platformnet.run import advance

GENERATED_DIR = Path(__file__).resolve().parent / "generated"
DEFAULT_LEVEL_PATH = LEVEL_PATH

MAX_X = GRID_W * TILE - PLAYER
MAX_Y = GRID_H * TILE - PLAYER
SIZE = PLAYER // TILE
TRIES = 1000
MAX_FIXED_MISSES = 100
TILE_TO_CHAR = {EMPTY: ".", SOLID: "#", EXIT: "E", SPIKE: "X"}




def is_free(state, level):
    return not any(level.overlaps(state, t) for t in (SOLID, SPIKE, EXIT))


def is_area_empty(tiles, col, row):
    for r in range(row, row + SIZE):
        for c in range(col, col + SIZE):
            if tiles[r][c] != EMPTY:
                return False
    return True


def clear_spots(tiles):
    last_row = GRID_H - SIZE
    last_col = GRID_W - SIZE

    spots = []
    for row in range(last_row + 1):
        for col in range(last_col + 1):
            if is_area_empty(tiles, col, row):
                spots.append((col, row))
    return spots


def tiles_to_text(tiles, start):
    grid = [[TILE_TO_CHAR[t] for t in row] for row in tiles.tolist()]
    c, r = start
    grid[r][c] = "S"
    return "\n".join("".join(row) for row in grid) + "\n"


def random_levels(count, rng):

    levels = []
    while len(levels) < count:
        for tiles in make_maps_np(count - len(levels), seed=rng.randrange(2**32)):
            spots = clear_spots(tiles)
            if not spots:
                continue
            text = tiles_to_text(tiles, rng.choice(spots))
            levels.append((text, parse_level(text)))
    return levels


def load_fixed_levels(level_path):

    if isinstance(level_path, (str, Path)):
        level_paths = [level_path]
    else:
        level_paths = list(level_path)

    paths = []
    levels = []
    for p in level_paths:
        path = resolve_level(p)
        text = path.read_text()
        level = load_level(path)

        paths.append(path)
        levels.append((text, level))
    return paths, levels


def random_air_state(level, rng):

    for _ in range(TRIES):
        state = (
            rng.uniform(0, MAX_X), rng.uniform(0, MAX_Y),
            rng.uniform(-MAX_RUN, MAX_RUN), rng.uniform(JUMP_VEL, MAX_FALL),
        )
        if is_free(state, level):
            return state
    return None


def random_grounded_state(level, rng):

    for _ in range(TRIES):
        x = rng.uniform(0, MAX_X)
        floor_row = rng.randrange(SIZE, GRID_H)
        y = float(floor_row * TILE - PLAYER)
        state = (x, y, rng.uniform(-MAX_RUN, MAX_RUN), 0.0)

        first_col = int(x // TILE)
        last_col = int(-(-(x + PLAYER) // TILE))
        on_floor = any(level.tile(c, floor_row) == SOLID for c in range(first_col, last_col))

        if on_floor and is_free(state, level):
            return state
    return None



def generate(count, level_path=DEFAULT_LEVEL_PATH, level_share=0.4, maps=100, grounded=0.5, seed=None,
             make_levels=random_levels):

    rng = random.Random(seed)

    paths, fixed = load_fixed_levels(level_path) if level_share > 0 else ([], [])
    randoms = make_levels(maps - len(fixed), rng) if level_share < 1 else []
    levels = fixed + randoms  

    pairs = []
    fixed_misses = 0

    while len(pairs) < count:

        use_fixed = rng.random() < level_share
        if use_fixed:
            index = rng.randrange(len(fixed))
        else:
            index = len(fixed) + rng.randrange(len(randoms))
        level = levels[index][1]

        if rng.random() < grounded:
            state = random_grounded_state(level, rng)
        else:
            state = random_air_state(level, rng)

        if state is None:
            if use_fixed:
                fixed_misses += 1
                if fixed_misses > MAX_FIXED_MISSES:
                    raise ValueError(f"No valid start position found in {paths[index]}")
            continue

        action = (rng.choice((-1, 0, 1)), rng.choice((0, 1)))
        _, dead, new_state, finished = advance(state, action, level)
        pairs.append(make_pair(index, state, action, new_state, dead, finished))

    level_texts = [text for text, _ in levels]
    return level_texts, pairs


def save(levels, pairs):
    """Write levels and pairs to a timestamped minified JSON file. Returns its path."""
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    path = GENERATED_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(json.dumps({"levels": levels, "pairs": pairs}, separators=(",", ":")))
    return path


def parse_args():
    parser = argparse.ArgumentParser(description="Generate pairs on a real level mixed with map_gen maps.")
    parser.add_argument("-n", "--count", type=int, default=250000, help="Number of pairs (default 250000)")
    parser.add_argument("--level", default=DEFAULT_LEVEL_PATH, help="Real level mixed in")
    parser.add_argument("--level-share", type=float, default=0.4, help="Fraction of pairs on real level (default 0.4)")
    parser.add_argument("--maps", type=int, default=100, help="Levels per file, including real level (default 100)")
    parser.add_argument("--grounded", type=float, default=0.5, help="Fraction starting on a floor (default 0.5)")
    parser.add_argument("--seed", type=int, help="Random seed for repeatable output")
    args = parser.parse_args()

    if not 0 <= args.level_share <= 1:
        parser.error("--level-share must be between 0 and 1")
    if 0 < args.level_share < 1 and args.maps < 2:
        parser.error("--maps must be at least 2 when mixing real levels with random maps")
    if args.level_share == 0 and args.maps < 1:
        parser.error("--maps must be at least 1")
    return args


if __name__ == "__main__":
    args = parse_args()
    levels, pairs = generate(args.count, args.level, args.level_share, args.maps, args.grounded, args.seed)
    path = save(levels, pairs)
    print(f"Generated {len(pairs)} pairs on {len(levels)} levels to {path}")
