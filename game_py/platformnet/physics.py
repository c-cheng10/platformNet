"""Ground-truth physics, ported from game/Game.java (updatePlayer). Pure Python,
no Pygame, so bots can run it headless: step(state, action, level) -> next_state.

Kept from the Java game (the feel): per-tick units at 100 ticks/s (Swing Timer
was 10 ms), ACCEL 0.7, multiplicative FRICTION 0.85 on the ground and in the air,
GRAVITY 0.3, JUMP_VEL -8, and jump-while-held (holding jump bunny-hops on landing).

Fixed for the neural net:
  1. Collisions snap to the contact surface (swept against the tile grid) instead
     of undoing the move, so resting positions are exact and history-independent.
  2. Fall speed is capped (Java's was unbounded), so states stay in a bounded range
     and nothing tunnels through thin walls.
  3. Hitbox = drawn sprite (Java drew a 40px sprite over a 5px hitbox).
  4. Touching an edge is not a collision (Java's >= / <= could stall you on a floor).
  5. grounded / on_ice are explicit state, tiny vx snaps to 0 for a clean rest state.
Teleports (death, room change) live in run.py and are flagged per frame.

Coordinates are room-local pixels, origin top-left, y down; (x, y) is the
player's top-left corner. Rooms are 100x100 tiles of 5px (a 500x500 screen).

State (tuple of 6 floats, booleans as 0.0/1.0): x, y, vx, vy, grounded, on_ice
Action (tuple of 2 ints): input_dir in {-1, 0, 1}, jump in {0, 1} (held)
No hidden timers or edge-triggered input: everything physics reads is in
(state, action). If you add physics that needs more, add it to the state.
"""
from math import ceil, floor

# Bump whenever constants or rules change: data recorded before is under different physics.
PHYSICS_VERSION = 1

TICK_HZ = 100
TILE = 5
PLAYER = 10
GRID_W = 100
GRID_H = 100

ACCEL = 0.7            # px/tick^2 while a direction is held (Java: 0.7)
FRICTION = 0.85        # vx multiplier every tick (Java: 0.85) -> top speed ~3.97 px/tick
ICE_FRICTION = 0.97    # replaces FRICTION on ice when no direction is held: long slides
GRAVITY = 0.3          # (Java: 0.3)
MAX_FALL = 10.0        # new: 2 tiles/tick
JUMP_VEL = -8.0        # (Java: -8) -> apex ~102 px
BOUNCE_VEL = 11.5      # bounce pad launch -> apex ~2x a jump
VX_EPS = 0.01          # |vx| below this snaps to 0

# Tile codes
EMPTY, SOLID, ICE, BOUNCE, ONE_WAY, EXIT, SPIKE = range(7)
BLOCKS_SIDE = (SOLID, ICE, BOUNCE)           # one-way never blocks sideways
BLOCKS_UP = (SOLID, ICE, BOUNCE)             # or from below
BLOCKS_DOWN = (SOLID, ICE, BOUNCE, ONE_WAY)  # one-way only via the swept "was above" test

STATE_DIM = 6
X, Y, VX, VY, GROUNDED, ON_ICE = range(STATE_DIM)
STATE_NAMES = ("x", "y", "vx", "vy", "grounded", "on_ice")


def step(state, action, level):
    """Advance one tick. `level` needs .tile(col, row) -> tile code."""
    x, y, vx, vy, _grounded, on_ice = state
    input_dir, jump = action

    vx += input_dir * ACCEL
    vx *= ICE_FRICTION if (on_ice > 0.5 and input_dir == 0) else FRICTION
    if -VX_EPS < vx < VX_EPS:
        vx = 0.0
    x, vx = _move_x(x, y, vx, level)

    vy = min(vy + GRAVITY, MAX_FALL)
    y, vy, grounded, on_ice = _move_y(x, y, vy, level)
    if grounded and jump:
        # Java decides the jump in the landing tick; it moves the player next tick.
        vy = JUMP_VEL
    return (x, y, vx, vy, float(grounded), float(on_ice))


def _move_x(x, y, vx, level):
    if vx == 0:
        return x, vx
    r0 = floor(y / TILE)
    r1 = ceil((y + PLAYER) / TILE) - 1
    if vx > 0:
        right = x + PLAYER
        for c in range(ceil(right / TILE), ceil((right + vx) / TILE)):
            for r in range(r0, r1 + 1):
                if level.tile(c, r) in BLOCKS_SIDE:
                    return float(c * TILE - PLAYER), 0.0
    else:
        for c in range(floor(x / TILE) - 1, floor((x + vx) / TILE) - 1, -1):
            for r in range(r0, r1 + 1):
                if level.tile(c, r) in BLOCKS_SIDE:
                    return float((c + 1) * TILE), 0.0
    return x + vx, vx


def _move_y(x, y, vy, level):
    """Returns (y, vy, grounded, on_ice)."""
    c0 = floor(x / TILE)
    c1 = ceil((x + PLAYER) / TILE) - 1
    if vy > 0:
        bottom = y + PLAYER
        # Only rows whose top is at or below the old bottom: this is the one-way
        # rule ("bottom was above the platform top last tick") in swept form.
        for r in range(ceil(bottom / TILE), ceil((bottom + vy) / TILE)):
            hit = ice = bounce = False
            for c in range(c0, c1 + 1):
                t = level.tile(c, r)
                if t in BLOCKS_DOWN:
                    hit = True
                    ice = ice or t == ICE
                    bounce = bounce or t == BOUNCE
            if hit:
                ny = float(r * TILE - PLAYER)
                if bounce:
                    return ny, -BOUNCE_VEL, False, False
                return ny, 0.0, True, ice
    elif vy < 0:
        for r in range(floor(y / TILE) - 1, floor((y + vy) / TILE) - 1, -1):
            for c in range(c0, c1 + 1):
                if level.tile(c, r) in BLOCKS_UP:
                    return float((r + 1) * TILE), 0.0, False, False
    return y + vy, vy, False, False
