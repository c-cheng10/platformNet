import argparse
import json
import random
import time
from pathlib import Path

from gamerecorder import make_pair
from platformnet.level import load_level, parse_level, resolve_level
from platformnet.physics import (
    EMPTY, EXIT, GRID_H, GRID_W, JUMP_VEL, MAX_FALL, MAX_RUN, PLAYER, SOLID, SPIKE, TILE
)
from platformnet.run import advance

GENERATED_DIR = Path(__file__).resolve().parent / "generated"
MAX_X = GRID_W * TILE - PLAYER
MAX_Y = GRID_H * TILE - PLAYER
SIZE = PLAYER // TILE
TRIES = 1000
TILE_TO_CHAR = {EMPTY: ".", SOLID: "#", EXIT: "E", SPIKE: "X"}

def count_solid_neighbors(grid, r, c):

    h = len(grid)
    w = len(grid[0])
    count = 0
    
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue 
                
            nr, nc = r + dr, c + dc
            
            if nr < 0 or nr >= h or nc < 0 or nc >= w:
                count += 1
            elif grid[nr][nc] == True:
                count += 1
                
    return count

def smooth_cave(grid, steps):
    """
    Applies the cave smoothing rule over a number of steps:
    - More than 4 wall neighbors -> becomes a wall (True)
    - Fewer than 4 wall neighbors -> becomes empty space (False)
    - Exactly 4 -> stays the same
    """
    h = len(grid)
    w = len(grid[0])
    
    for _ in range(steps):
        new_grid = [[False for _ in range(w)] for _ in range(h)]
        
        for r in range(h):
            for c in range(w):
                walls = count_solid_neighbors(grid, r, c)
                if walls > 4:
                    new_grid[r][c] = True
                elif walls < 4:
                    new_grid[r][c] = False
                else:
                    new_grid[r][c] = grid[r][c]
                    
        grid = new_grid 
    return grid


def generate_cave_map(rng):

    h, w = GRID_H, GRID_W
    
    grid = [[rng.random() < 0.45 for _ in range(w)] for _ in range(h)]
    
    grid = smooth_cave(grid, 5)
    
    map_tiles = []
    for r in range(h):
        row = []
        for c in range(w):
            if grid[r][c]:
                row.append(SOLID)
            else:
                row.append(EMPTY)
        map_tiles.append(row)
        
    return map_tiles


def add_spikes(m, rng):
    h = len(m)
    w = len(m[0])
    
    for r in range(h - 1):
        for c in range(w):
            if m[r][c] == EMPTY and m[r + 1][c] == SOLID:
                if rng.random() < 0.05:
                    length = rng.randint(1, 5)
                    for i in range(length):
                        nc = c + i
                        if nc < w and m[r][nc] == EMPTY and m[r + 1][nc] == SOLID:
                            m[r][nc] = SPIKE


def add_exit(m, rng):

    h = len(m)
    w = len(m[0])

    box_w = rng.randint(2, 5)
    box_h = rng.randint(2, 5)
    start_r = rng.randint(0, h - box_h)
    start_c = rng.randint(0, w - box_w)
    
    for r in range(start_r, start_r + box_h):
        for c in range(start_c, start_c + box_w):
            if m[r][c] == EMPTY:
                m[r][c] = EXIT


def make_clumped_map(rng):

    m = generate_cave_map(rng)
    add_spikes(m, rng)
    add_exit(m, rng)
    return m


def clumped_levels(count, rng):

    levels = []
    while len(levels) < count:
        tiles = make_clumped_map(rng)
        
        clear_spots = []
        h = len(tiles)
        w = len(tiles[0])
        
        for r in range(h - SIZE + 1):
            for c in range(w - SIZE + 1):
                all_empty = True
                for pr in range(SIZE):
                    for pc in range(SIZE):
                        if tiles[r + pr][c + pc] != EMPTY:
                            all_empty = False
                if all_empty:
                    clear_spots.append((c, r))
                    
        if not clear_spots:
            continue 

        grid_chars = []
        for row in tiles:
            char_row = [TILE_TO_CHAR[t] for t in row]
            grid_chars.append(char_row)
            
        spawn_c, spawn_r = rng.choice(clear_spots)
        grid_chars[spawn_r][spawn_c] = "S"
        
        text = "\n".join("".join(row) for row in grid_chars) + "\n"
        levels.append((text, parse_level(text)))
        
    return levels

def is_reachable(state, level):

    return not any(level.overlaps(state, t) for t in (SOLID, SPIKE, EXIT))


def random_air_state(level, rng):
 
    for _ in range(TRIES):
        state = (
            rng.uniform(0, MAX_X), rng.uniform(0, MAX_Y),
            rng.uniform(-MAX_RUN, MAX_RUN), rng.uniform(JUMP_VEL, MAX_FALL)
        )
        if is_reachable(state, level):
            return state
    return None


def random_grounded_state(level, rng):

    for _ in range(TRIES):
        x = rng.uniform(0, MAX_X)
        floor_row = rng.randint(SIZE, GRID_H - 1)
        state = (x, float(floor_row * TILE - PLAYER), rng.uniform(-MAX_RUN, MAX_RUN), 0.0)
        
        c_start = int(x // TILE)
        c_end = int((x + PLAYER) // TILE)
        has_floor = False
        for c in range(c_start, c_end + 1):
            if level.tile(c, floor_row) == SOLID:
                has_floor = True
                
        if has_floor and is_reachable(state, level):
            return state
    return None

def generate(count, level_path=None, level_share=0.4, maps=100, grounded=0.5, seed=None):
   
    rng = random.Random(seed)
    fixed = []
    
    if level_share > 0:
        level_file = resolve_level(level_path)
        fixed = [(level_file.read_text(), load_level(level_file))]
        
    n_random = maps - len(fixed) if level_share < 1 else 0
    levels = fixed + clumped_levels(n_random, rng)
    
    pairs = []
    while len(pairs) < count:
        use_fixed = bool(fixed) and (not n_random or rng.random() < level_share)
        index = 0 if use_fixed else len(fixed) + rng.randrange(n_random)
        level = levels[index][1]
        
        if rng.random() < grounded:
            state = random_grounded_state(level, rng)
        else:
            state = random_air_state(level, rng)
            
        if state is None:
            continue
            
        action = (rng.choice((-1, 0, 1)), rng.choice((0, 1)))
        _, dead, new_state, finished = advance(state, action, level)
        pairs.append(make_pair(index, state, action, new_state, dead, finished))
        
    return [text for text, _ in levels], pairs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate simple cave training data.")
    parser.add_argument("-n", "--count", type=int, default=1000, help="Number of pairs to generate")
    parser.add_argument("--level", default=None, help="Path to a reference level file")
    parser.add_argument("--level-share", type=float, default=0.4, help="Fraction of pairs using reference level")
    parser.add_argument("--maps", type=int, default=20, help="Total maps to use")
    parser.add_argument("--grounded", type=float, default=0.5, help="Fraction starting on floor")
    parser.add_argument("--seed", type=int, help="Random seed")
    args = parser.parse_args()

    levels, pairs = generate(args.count, args.level, args.level_share, args.maps, args.grounded, args.seed)
    
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    path = GENERATED_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}-clumped.json"
    path.write_text(json.dumps({"levels": levels, "pairs": pairs}, separators=(",", ":")))
    
    print(f"Generated {len(pairs)} pairs on {len(levels)} levels and saved to {path}")