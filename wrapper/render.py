import pygame
import sys
import os
import random
from pathlib import Path

LEVELS_DIR = Path(__file__).resolve().parents[1] / "levels"
ART_DIR = Path(__file__).resolve().parents[1] / "art"
BG_ART_DIR = ART_DIR / "bg"
MODEL_PATH = Path(__file__).resolve().parent / "model_b128_e100_noise2_p30.pt"

# Because of small errors the model can sometimes be pushed a tiny bit into the ground, which ends
# up making it sink. This basically forces it to stay on top. Press F to toggle.
SNAP_TO_FLOOR = False
SNAP_DIST = 1.0    # px: how far above/into the floor still counts as standing on it
SNAP_MIN_VY = -1.0  # px/tick: never snap while moving up faster than this (taking off from a jump)

# The model and its pre/post-processing live in neuralnet/train.py
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "neuralnet"))
from train import run_model

# Initialize Pygame
pygame.init()

### CONFIGURATION ###
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 1000
TILE_SIZE = 10 # Should be SCREEN_WIDTH/100

# 5px tiles, 10px player
GAME_TILE = 5
GAME_PLAYER = 10
SCALE = TILE_SIZE / GAME_TILE  # game px -> screen px
LEVEL_PX = 100 * GAME_TILE  # levels are 100x100 tiles

screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Platformer Renderer")
clock = pygame.time.Clock()

### COLORS ###
COLOR_BG = (0xf2, 0xf2, 0xf2)   # Empty space (.) fallback
COLOR_GROUND = (0, 0, 0)        # Ground blocks (#)
#COLOR_START = (46, 204, 113)   # Start position (S)
COLOR_EXIT = (241, 196, 15)     # Exit (E)
COLOR_SPIKE = (231, 76, 60)     # Spike (X)
COLOR_PLAYER = (255, 255, 255)  # Fallback player visual

import pygame

def adjust_saturation(surface, saturation_factor):
    """
    Changes the saturation of a Pygame surface. Scale of 0.0-1.0.
    """
    # 1. Copy the original surface so we don't mutate the source
    result = surface.copy()
    
    # If factor is 1.0, no changes needed
    if saturation_factor >= 1.0:
        return result
        
    # 2. Generate the built-in grayscale version of the surface
    gray_surface = pygame.transform.grayscale(surface)
    
    # 3. Calculate alpha for the grayscale overlay 
    # (0.0 saturation = 255 alpha overlay, 1.0 saturation = 0 alpha overlay)
    alpha_value = int((1.0 - saturation_factor) * 255)
    gray_surface.set_alpha(alpha_value)
    
    # 4. Blit the semi-transparent grayscale image over the copy to wash out colors
    result.blit(gray_surface, (0, 0))
    
    return result

def load_level(level_num):
    """Loads level data from a file given the level number."""
    filepath = LEVELS_DIR / f"level{level_num}.txt"
    if not os.path.exists(filepath):
        return None, None
    
    with open(filepath, "r") as f:
        level_data = f.read()
        
    level_grid = [list(row) for row in level_data.strip().splitlines()]
    return level_data, level_grid

def load_random_player_art():
    """Finds all image files in /art and picks one randomly, also scales it to the right size."""
    if not ART_DIR.exists():
        return None
    
    art_files = (
        list(ART_DIR.glob("*.png")) + list(ART_DIR.glob("*.jpg")) + list(ART_DIR.glob("*.jpeg")) +
        list(ART_DIR.glob("*.PNG")) + list(ART_DIR.glob("*.JPG")) + list(ART_DIR.glob("*.JPEG"))
    )
    if not art_files:
        return None
    
    chosen_file = random.choice(art_files)
    try:
        img = pygame.image.load(str(chosen_file)).convert_alpha()
        # Scale to player dimensions (2x2 tiles)
        player_size = 2 * TILE_SIZE
        return pygame.transform.smoothscale(img, (player_size, player_size))
    except Exception as e:
        print(f"Error loading player art {chosen_file}: {e}")
        return None

def load_random_background():
    """Finds all image files in /art/bg and picks one randomly."""
    if not BG_ART_DIR.exists():
        return None
    
    bg_files = (
        list(BG_ART_DIR.glob("*.png")) + list(BG_ART_DIR.glob("*.jpg")) + list(BG_ART_DIR.glob("*.jpeg")) +
        list(BG_ART_DIR.glob("*.PNG")) + list(BG_ART_DIR.glob("*.JPG")) + list(BG_ART_DIR.glob("*.JPEG"))
    )
    if not bg_files:
        return None
    
    chosen_file = random.choice(bg_files)
    try:
        img = pygame.image.load(str(chosen_file)).convert()
        return adjust_saturation(pygame.transform.smoothscale(img, (SCREEN_WIDTH, SCREEN_HEIGHT)), 0.5)
    except Exception as e:
        print(f"Error loading background art {chosen_file}: {e}")
        return None

def get_start_position(grid):
    """Finds the starting tile coordinates."""
    for r_idx, row in enumerate(grid):
        if 'S' in row:
            return [float(row.index('S') * GAME_TILE), float(r_idx * GAME_TILE)]
    return [0.0, 0.0]

def render_level(surface, grid, bg_img):
    """Draws the level."""
    if bg_img:
        #surface.blit(adjust_saturation(bg_img, saturation_factor=0.5), (0, 0))
        surface.blit(bg_img, (0, 0))
    else:
        surface.fill(COLOR_BG)
        
    for i, row in enumerate(grid):
        for j, tile in enumerate(row):
            if tile == '.':
                continue
            rect = pygame.Rect(j * TILE_SIZE, i * TILE_SIZE, TILE_SIZE, TILE_SIZE)
            if tile == '#':
                pygame.draw.rect(surface, COLOR_GROUND, rect)
            #elif tile == 'S':
            #    pygame.draw.rect(surface, COLOR_START, rect)
            elif tile == 'E':
                pygame.draw.rect(surface, COLOR_EXIT, rect)
            elif tile == 'X':
                pygame.draw.rect(surface, COLOR_SPIKE, rect)

def render_player(surface, pos, player_img):
    """Renders the player on screen. pos is in game px."""
    x = int(pos[0] * SCALE)
    y = int(pos[1] * SCALE)
    if player_img:
        surface.blit(player_img, (x, y))
    else:
        # fall back to using a rectangle as the player if the art didn't load
        player_rect = pygame.Rect(x, y, 2*TILE_SIZE, 2*TILE_SIZE)
        pygame.draw.rect(surface, COLOR_PLAYER, player_rect)

def overlaps(level, pos, char):
    """checks if the player strictly overlaps any tile."""
    rows = [line for line in level.splitlines() if line.strip() and not line.startswith(";")]
    x, y = pos
    for r in range(int(y // GAME_TILE), int(-(-(y + GAME_PLAYER) // GAME_TILE))):
        for c in range(int(x // GAME_TILE), int(-(-(x + GAME_PLAYER) // GAME_TILE))):
            if 0 <= r < len(rows) and 0 <= c < len(rows[r]) and rows[r][c] == char:
                return True
    return False

# THIS FUNCTION IS NOT USED.
#def snap_to_floor(level, out):
#    """If the predicted player is within SNAP_DIST of resting on solid ground, put it exactly there with vy = 0."""
#    if out["vely"] < SNAP_MIN_VY:
#        return
#    rows = [line for line in level.splitlines() if line.strip() and not line.startswith(";")]
#    x, y = out["posx"], out["posy"]
#    r = round((y + GAME_PLAYER) / GAME_TILE)  # tile row nearest the player's bottom edge
#    if abs((y + GAME_PLAYER) - r * GAME_TILE) > SNAP_DIST:
#        return
#    cols = range(int(x // GAME_TILE), int(-(-(x + GAME_PLAYER) // GAME_TILE)))
#    # Anything outside the grid counts as solid, as in the game
#    if any(not (0 <= r < len(rows) and 0 <= c < len(rows[r])) or rows[r][c] == "#" for c in cols):
#        out["posy"] = float(r * GAME_TILE - GAME_PLAYER)
#        out["vely"] = 0.0


def get_next_frame_raw(data):
    """
    Gets the neural network's decisions based on inputs and return it.
    """
    out = run_model(data, path=MODEL_PATH)
    #if SNAP_TO_FLOOR:
    #    snap_to_floor(data["LEVEL"], out)
    # The NN only predicts motion, it doesn't calculate death or if the level is finished.
    pos = (out["posx"], out["posy"])
    out["isDead"] = int(overlaps(data["LEVEL"], pos, "X"))
    out["isFinished"] = int(not out["isDead"] and overlaps(data["LEVEL"], pos, "E"))
    return out

def get_next_frame(right_pressed, left_pressed, up_pressed, position, velocity, level):
    data = get_next_frame_raw({
                           "shouldRight": int(right_pressed),
                           "shouldLeft": int(left_pressed),
                           "wantsJump": int(up_pressed),
                           "LEVEL": level,
                           "prevposx": position[0],
                           "prevposy": position[1],
                           "prevvelx": velocity[0],
                           "prevvely": velocity[1],
                       })
    new_pos = [data.get("posx", position[0]), data.get("posy", position[1])]
    new_vel = [data.get("velx", velocity[0]), data.get("vely", velocity[1])]
    # Keep the player on the screen.
    for i in range(2):
        clamped = min(max(new_pos[i], 0.0), LEVEL_PX - GAME_PLAYER)
        if clamped != new_pos[i]:
            new_pos[i], new_vel[i] = clamped, 0.0
    is_dead = bool(data.get("isDead", False))
    is_finished = bool(data.get("isFinished", False))
    
    return new_pos, new_vel, is_dead, is_finished

### MAIN LOOP ###
def main():
    global SNAP_TO_FLOOR
    current_level = 1
    LEVEL_DATA, level_grid = load_level(current_level)
    
    if not LEVEL_DATA:
        print(f"Error: Could not find levels/level{current_level}.txt")
        sys.exit()

    # Load random player and background images.
    player_img = load_random_player_art()
    bg_img = load_random_background()

    start_pos = get_start_position(level_grid)
    player_pos = list(start_pos)
    player_vel = [0.0, 0.0]
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            #if event.type == pygame.KEYDOWN and event.key == pygame.K_f:
            #    SNAP_TO_FLOOR = not SNAP_TO_FLOOR
            #    print(f"Floor snapping {'on' if SNAP_TO_FLOOR else 'off'}")

        # 1-3. Check key states
        keys = pygame.key.get_pressed()
        right_pressed = keys[pygame.K_RIGHT]
        left_pressed = keys[pygame.K_LEFT]
        up_pressed = keys[pygame.K_UP]
        reset_pressed = keys[pygame.K_r]

        # 4 & 5. Pass inputs, position, and velocity into get_next_frame
        player_pos, player_vel, is_dead, is_finished = get_next_frame(
            right_pressed,
            left_pressed,
            up_pressed,
            player_pos,
            player_vel,
            LEVEL_DATA,
        )

        # Handle player death / level reset
        if is_dead or reset_pressed:
            player_img = load_random_player_art()
            bg_img = load_random_background()
            player_pos = list(start_pos)
            player_vel = [0.0, 0.0]

        # Handle level completion / loading the next level
        if is_finished:
            player_img = load_random_player_art()
            bg_img = load_random_background()
            print(f"Level {current_level} Complete!")
            current_level += 1
            next_data, next_grid = load_level(current_level)
            
            if next_data:
                # Load next level
                LEVEL_DATA = next_data
                level_grid = next_grid
                start_pos = get_start_position(level_grid)
                player_pos = list(start_pos)
                player_vel = [0.0, 0.0]
                print(f"Loading Level {current_level}...")
            else:
                print("You beat all the levels! Restarting game from Level 1.")
                current_level = 1
                LEVEL_DATA, level_grid = load_level(current_level)
                start_pos = get_start_position(level_grid)
                player_pos = list(start_pos)
                player_vel = [0.0, 0.0]

        # Rendering
        render_level(screen, level_grid, bg_img)
        render_player(screen, player_pos, player_img)

        pygame.display.flip()
        clock.tick(60)  # the model was trained on 60 fps physics

if __name__ == "__main__":
    main()
