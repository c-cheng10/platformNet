import pygame
import sys

# Initialize Pygame
pygame.init()

# --- Configuration ---
SCREEN_WIDTH = 500
SCREEN_HEIGHT = 500
TILE_SIZE = 5  # 100x100 grid fits a 500x500 screen perfectly

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

# --- Level Data ---
with open("levels/level1.txt", "r") as f:
    LEVEL_DATA = f.read()

# Parse level grid
level_grid = [list(row) for row in LEVEL_DATA.strip().splitlines()]
LEVEL_HEIGHT = len(level_grid)
LEVEL_WIDTH = len(level_grid[0]) if LEVEL_HEIGHT > 0 else 0

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
    player_rect = pygame.Rect(int(pos[0]), int(pos[1]), TILE_SIZE, TILE_SIZE)
    pygame.draw.rect(surface, COLOR_PLAYER, player_rect)

# --- Your Physics Function ---
def get_next_frame(right_pressed, left_pressed, up_pressed, position, velocity, level):
    new_pos = position
    new_vel = velocity
    is_dead = False
    
    return new_pos, new_vel, is_dead

# --- Main Game Loop ---
def main():
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
        player_pos, player_vel, is_dead = get_next_frame(
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

        # Rendering
        render_level(screen, level_grid)
        render_player(screen, player_pos)

        pygame.display.flip()
        clock.tick(60)

if __name__ == "__main__":
    main()
