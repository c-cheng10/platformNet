import argparse
import json
import random
import time
from pathlib import Path

import numpy as np

from gamerecorder import make_pair
from platformnet.level import load_level, parse_level, resolve_level
from platformnet.physics import (
    EMPTY, EXIT, GRID_H, GRID_W, JUMP_VEL, MAX_FALL, MAX_RUN, PLAYER, SOLID, SPIKE, TILE
)
from platformnet.run import advance

# --- DIRECTORIES & CONSTANTS ---
GENERATED_DIR = Path(__file__).resolve().parent / "generated"
MAX_X = GRID_W * TILE - PLAYER
MAX_Y = GRID_H * TILE - PLAYER
SIZE = PLAYER // TILE  # The player's size measured in tiles (e.g., 2x2)
TRIES = 1000           # How many times to try placing something before giving up
TILE_TO_CHAR = {EMPTY: ".", SOLID: "#", EXIT: "E", SPIKE: "X"}


# ==============================================================================
# SECTION 1: PROCEDURAL CAVE GENERATION (Cellular Automata)
# ==============================================================================

def _solid_neighbours(solid):
    """
    Counts how many solid wall tiles exist around each tile in its 8 neighboring spots.
    Uses numpy padding so that tiles on the outer edge of the map treat the 
    outside world as solid walls.
    """
    # Pad the grid with 'True' (solid) all around the borders
    padded_grid = np.pad(solid, 1, constant_values=True).astype(np.int8)
    h, w = solid.shape
    
    # Sum up all 8 surrounding shifts (diagonal, horizontal, and vertical offsets)
    return sum(
        padded_grid[1 + dr : 1 + dr + h, 1 + dc : 1 + dc + w]
        for dr in (-1, 0, 1) 
        for dc in (-1, 0, 1) 
        if dr or dc  # Skip the center tile itself (dr=0 and dc=0)
    )


def _smooth(solid, steps):
    """
    Applies the classic Cave/Smoothing rule over a number of steps:
    - If a tile has MORE than 4 solid neighbors, it becomes a solid wall.
    - If it has FEWER than 4 solid neighbors, it becomes empty air.
    - If it has EXACTLY 4, it stays unchanged.
    This turns random noise into smooth, organic cave structures and tunnels.
    """
    for _ in range(steps):
        neighbors = _solid_neighbours(solid)
        solid = (neighbors > 4) | (solid & (neighbors == 4))
    return solid


def _caves(rng):
    """
    Generates a random boolean grid where True = solid wall, False = empty space.
    Uses scaling to create either small detailed tunnels or big chunky rooms.
    """
    # Choose a scale: 1 = normal grid, 2 or 4 = chunky/coarse grid (bigger rooms)
    scale = rng.choice((1, 1, 2, 2, 4))  
    h, w = -(-GRID_H // scale), -(-GRID_W // scale)
    
    # Randomly fill a percentage of the coarse grid with walls (between 38% and 52%)
    fill_density = rng.uniform(0.38, 0.52)
    initial_noise = rng.random((h, w)) < fill_density
    
    # Smooth the noise into cave shapes
    solid = _smooth(initial_noise, rng.integers(3, 7))
    
    # Upscale the coarse grid back to the full game dimensions using kronecker product
    solid = np.kron(solid, np.ones((scale, scale), dtype=bool))[:GRID_H, :GRID_W]
    
    # If we upscaled, rough up the blocky edges slightly and smooth them back out
    if scale > 1:
        flip_noise = rng.random(solid.shape) < 0.08
        solid = _smooth(solid ^ flip_noise, 2)
        
    return solid


def _add_spikes(m, rng):
    """Scatters short runs of deadly spikes on top of flat horizontal floors."""
    # Find spots where an empty tile sits directly on top of a solid floor tile
    rows, cols = np.nonzero((m[:-1] == EMPTY) & (m[1:] == SOLID))
    if not len(rows):
        return
    
    # Randomly pick a few valid locations to place runs of spikes
    num_spikes = min(len(rows), int(rng.integers(0, 14)))
    for i in rng.choice(len(rows), size=num_spikes, replace=False):
        r, c = rows[i], cols[i]
        segment_length = int(rng.integers(1, 10))
        seg = slice(c, min(c + segment_length, GRID_W))
        
        # Turn empty floor spots into spikes
        valid_spots = (m[r, seg] == EMPTY) & (m[r + 1, seg] == SOLID)
        m[r, seg][valid_spots] = SPIKE


def _add_exit(m, rng):
    """Places goal/exit regions ('E') inside open pockets of the cave."""
    for _ in range(rng.choice((0, 1, 1, 1, 2))):
        w, h = int(rng.integers(2, 7)), int(rng.integers(2, 10))
        r, c = int(rng.integers(0, GRID_H - h + 1)), int(rng.integers(0, GRID_W - w + 1))
        
        # Replace empty air tiles inside this box with exit tiles
        box = m[r:r + h, c:c + w]
        box[box == EMPTY] = EXIT


def make_clumped_map(rng):
    """Combines caves, spikes, and exits into a complete 2D level grid."""
    # Convert boolean cave grid into integers (SOLID vs EMPTY)
    m = np.where(_caves(rng), SOLID, EMPTY)
    _add_spikes(m, rng)
    _add_exit(m, rng)
    return m


def clumped_levels(count, rng):
    """
    Generates multiple valid levels, ensuring each has at least one clear 
    spawn location 'S' large enough for the player to fit into.
    """
    np_rng = np.random.default_rng(rng.randrange(2**32))
    levels = []
    
    while len(levels) < count:
        tiles = make_clumped_map(np_rng)
        empty = tiles == EMPTY
        
        # Find all clear spots on the grid where the player can spawn without colliding with walls
        clear_spots = [
            (c, r) for r in range(GRID_H - SIZE + 1) for c in range(GRID_W - SIZE + 1)
            if empty[r:r + SIZE, c:c + SIZE].all()
        ]
        
        if not clear_spots:
            continue  # Try again if no valid space is found
            
        # Convert tile IDs to characters
        grid = [[TILE_TO_CHAR[t] for t in row] for row in tiles.tolist()]
        
        # Place the player spawn point 'S' at a randomly chosen clear location
        c, r = rng.choice(clear_spots)
        grid[r][c] = "S"
        
        # Format the level as a plain text string
        text = "\n".join("".join(row) for row in grid) + "\n"
        levels.append((text, parse_level(text)))
        
    return levels


# ==============================================================================
# SECTION 2: GAME STATE SIMULATION & DATA SAMPLING
# ==============================================================================

def is_reachable(state, level):
    """Checks if the player is alive and playing (not stuck in a wall, dead on spikes, or finished)."""
    return not any(level.overlaps(state, t) for t in (SOLID, SPIKE, EXIT))


def random_air_state(level, rng):
    """Spawns the player floating in mid-air at a valid random coordinate."""
    for _ in range(TRIES):
        state = (
            rng.uniform(0, MAX_X), 
            rng.uniform(0, MAX_Y),
            rng.uniform(-MAX_RUN, MAX_RUN), 
            rng.uniform(JUMP_VEL, MAX_FALL)
        )
        if is_reachable(state, level):
            return state
    return None


def random_grounded_state(level, rng):
    """Spawns the player standing securely on top of a solid floor tile."""
    for _ in range(TRIES):
        x = rng.uniform(0, MAX_X)
        floor_row = rng.randrange(SIZE, GRID_H)
        state = (x, float(floor_row * TILE - PLAYER), rng.uniform(-MAX_RUN, MAX_RUN), 0.0)
        
        # Verify there is actually a solid tile underneath the player
        under_tiles = range(int(x // TILE), int(-(-(x + PLAYER) // TILE)))
        if any(level.tile(c, floor_row) == SOLID for c in under_tiles) and is_reachable(state, level):
            return state
    return None


def generate(count, level_path=None, level_share=0.4, maps=100, grounded=0.5, seed=None):
    """
    Main data generation pipeline: mixes hand-crafted levels with procedural cave levels,
    simulates player actions, and pairs initial states with resulting outcomes.
    """
    rng = random.Random(seed)
    fixed = []
    
    # Load a preset/real level if specified
    if level_share > 0:
        level_file = resolve_level(level_path)
        fixed = [(level_file.read_text(), load_level(level_file))]
        
    n_random = maps - len(fixed) if level_share < 1 else 0
    levels = fixed + clumped_levels(n_random, rng)
    
    pairs = []
    fixed_misses = 0
    
    while len(pairs) < count:
        # Decide whether to use the fixed real level or a procedural cave level
        use_fixed = bool(fixed) and (not n_random or rng.random() < level_share)
        index = 0 if use_fixed else len(fixed) + rng.randrange(n_random)
        level = levels[index][1]
        
        # Spawn player either grounded on a floor or floating in air
        if rng.random() < grounded:
            state = random_grounded_state(level, rng)
        else:
            state = random_air_state(level, rng)
            
        if state is None:
            fixed_misses += use_fixed
            if fixed_misses > 100:
                raise ValueError(f"Could not find a valid start position in level {level_file}")
            continue
            
        # Pick a random control action for the player (movement direction + jump input)
        action = (rng.choice((-1, 0, 1)), rng.choice((0, 1)))
        
        # Simulate one physics step forward
        _, dead, new_state, finished = advance(state, action, level)
        pairs.append(make_pair(index, state, action, new_state, dead, finished))
        
    return [text for text, _ in levels], pairs


# ==============================================================================
# SECTION 3: COMMAND LINE INTERFACE (CLI)
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate clumped cave platformer training data.")
    parser.add_argument("-n", "--count", type=int, default=250000, help="Number of training pairs to generate")
    parser.add_argument("--level", default=None, help="Path to a real reference level to mix in")
    parser.add_argument("--level-share", type=float, default=0.4, help="Fraction of pairs using the reference level")
    parser.add_argument("--maps", type=int, default=100, help="Total number of unique maps to use")
    parser.add_argument("--grounded", type=float, default=0.5, help="Fraction of samples starting on a floor")
    parser.add_argument("--seed", type=int, help="Random seed for reproducible results")
    args = parser.parse_args()

    # Validate inputs
    if not 0 <= args.level_share <= 1:
        parser.error("--level-share must be between 0 and 1")
    if 0 < args.level_share < 1 and args.maps < 2:
        parser.error("--maps must be at least 2 when mixing levels")

    # Run the generator
    levels, pairs = generate(
        args.count, args.level, args.level_share, args.maps, args.grounded, args.seed
    )
    
    # Save the output as a minified JSON file
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    path = GENERATED_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}-clumped.json"
    path.write_text(json.dumps({"levels": levels, "pairs": pairs}, separators=(",", ":")))
    
    print(f"Successfully generated {len(pairs)} pairs across {len(levels)} levels and saved to {path}")