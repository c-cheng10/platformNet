import pygame
import sys
import os

# Initialize Pygame
pygame.init()

# --- Configuration ---
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 1000
TILE_SIZE = 10 # 100x100 grid fits a 500x500 screen perfectly

screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Platformer Renderer")
clock = pygame.time.Clock()

# --- Colors ---
COLOR_BG = (20, 20, 30)         # Empty space (.)
COLOR_GROUND = (70, 130, 180)   # Ground blocks (#)
COLOR_START = (46, 204, 113)    # Start position (S)
COLOR_EXIT = (241, 196, 15)     # Exit (E)
COLOR_SPIKE = (231, 76, 60)     # Spike (X)
COLOR_PLAYER = (255, 255, 255)  # Player visual

def load_level(level_num):
    """Loads level data from a file based on level number."""
    filepath = f"levels/level{level_num}.txt"
    if not os.path.exists(filepath):
        return None, None
    
    with open(filepath, "r") as f:
        level_data = f.read()
        
    level_grid = [list(row) for row in level_data.strip().splitlines()]
    return level_data, level_grid

def get_start_position(grid):
    """Finds the 'S' starting tile coordinates."""
    for r_idx, row in enumerate(grid):
        if 'S' in row:
            return [float(row.index('S') * TILE_SIZE), float(r_idx * TILE_SIZE)]
    return [0.0, 0.0]

def render_level(surface, grid):
    """Draws the full level grid."""
    surface.fill(COLOR_BG)
    for row_idx, row in enumerate(grid):
        for col_idx, tile in enumerate(row):
            if tile == '.':
                continue
            rect = pygame.Rect(col_idx * TILE_SIZE, row_idx * TILE_SIZE, TILE_SIZE, TILE_SIZE)
            if tile == '#':
                pygame.draw.rect(surface, COLOR_GROUND, rect)
            elif tile == 'S':
                pygame.draw.rect(surface, COLOR_START, rect)
            elif tile == 'E':
                pygame.draw.rect(surface, COLOR_EXIT, rect)
            elif tile == 'X':
                pygame.draw.rect(surface, COLOR_SPIKE, rect)

def render_player(surface, pos):
    """Renders the player on screen."""
    player_rect = pygame.Rect(int(pos[0]), int(pos[1]), 2*TILE_SIZE, 2*TILE_SIZE)
    pygame.draw.rect(surface, COLOR_PLAYER, player_rect)

def get_next_frame_raw(data):
    return {}

def get_next_frame(right_pressed, left_pressed, up_pressed, position, velocity, level):
    data = get_next_frame_raw({
                           "shouldRight": 0,
                           "shouldLeft": 0,
                           "wantsJump": 1,
                           "LEVEL": level,
                           "prevposx": position[0],
                           "prevposy": position[1],
                           "prevvelx": velocity[0],
                           "prevvely": velocity[1],
                       })
    new_pos = (data.get("posx", position[0]), data.get("posy", position[1]))
    new_vel = (data.get("velx", velocity[0]), data.get("vely", velocity[1]))
    is_dead = bool(data.get("isDead", False))
    is_finished = bool(data.get("isFinished", False))
    
    return new_pos, new_vel, is_dead, is_finished

# --- Main Game Loop ---
def main():
    current_level = 1
    LEVEL_DATA, level_grid = load_level(current_level)
    
    if not LEVEL_DATA:
        print(f"Error: Could not find levels/level{current_level}.txt")
        sys.exit()

    start_pos = get_start_position(level_grid)
    player_pos = list(start_pos)
    player_vel = [0.0, 0.0]
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # 1-3. Check key states
        keys = pygame.key.get_pressed()
        right_pressed = keys[pygame.K_RIGHT]
        left_pressed = keys[pygame.K_LEFT]
        up_pressed = keys[pygame.K_UP]

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
        if is_dead:
            player_pos = list(start_pos)
            player_vel = [0.0, 0.0]

        # Handle level completion / loading the next level
        if is_finished:
            print(f"Level {current_level} Complete!")
            current_level += 1
            next_data, next_grid = load_level(current_level)
            
            if next_data:
                # Load next level successfully
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
        render_level(screen, level_grid)
        render_player(screen, player_pos)

        pygame.display.flip()
        clock.tick(30)

if __name__ == "__main__":
    main()
