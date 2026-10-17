"""Ground-truth physics, ported from game/Game.java (updatePlayer). Pure Python,
no Pygame, so bots can run it headless: step(state, action, level) -> next_state.

Deliberately minimal: acceleration, friction, fixed-height jump, gravity, solid
collision. Kill blocks (spikes) are handled in run.py.

Kept from the Java game: per-tick units at 100 ticks/s (Swing Timer was 10 ms),
ACCEL 0.7, multiplicative FRICTION 0.85, GRAVITY 0.3, JUMP_VEL -8, jump-while-held.

Fixed for the neural net:
  1. Collisions snap to the contact surface (swept against the tile grid) instead
     of undoing the move, so resting positions are exact and history-independent.
  2. Fall speed is capped (Java's was unbounded): bounded states, no tunnelling.
  3. Hitbox = drawn sprite (Java drew a 40px sprite over a 5px hitbox).
  4. Touching an edge is not a collision.
  5. grounded is explicit state; tiny vx snaps to 0 for a clean rest state.

Coordinates are pixels, origin top-left, y down; (x, y) is the player's top-left.
The level is 100x100 tiles of 5px (a 500x500 screen).

State (tuple of 5 floats): x, y, vx, vy, grounded (0.0/1.0)
Action (tuple of 2 ints): input_dir in {-1, 0, 1}, jump in {0, 1} (held)
"""
from math import ceil, floor

# Bump whenever constants or rules change: data recorded before is under different physics.
PHYSICS_VERSION = 2

TICK_HZ = 100
TILE = 5
PLAYER = 10
GRID_W = 100
GRID_H = 100

ACCEL = 0.7            # (Java: 0.7)
FRICTION = 0.85        # vx multiplier every tick (Java: 0.85) -> top speed ~3.97 px/tick
GRAVITY = 0.3          # (Java: 0.3)
MAX_FALL = 10.0        # new: 2 tiles/tick
JUMP_VEL = -8.0        # (Java: -8) -> apex ~103 px
VX_EPS = 0.01          # |vx| below this snaps to 0

# Tile codes
EMPTY, SOLID, EXIT, SPIKE = range(4)

STATE_DIM = 5
X, Y, VX, VY, GROUNDED = range(STATE_DIM)
STATE_NAMES = ("x", "y", "vx", "vy", "grounded")


def step(state, action, level):
    """Advance one tick. `level` needs .tile(col, row) -> tile code."""
    x, y, vx, vy, _grounded = state
    input_dir, jump = action

    vx = (vx + input_dir * ACCEL) * FRICTION
    if -VX_EPS < vx < VX_EPS:
        vx = 0.0
    x, vx = _move_x(x, y, vx, level)

    vy = min(vy + GRAVITY, MAX_FALL)
    y, vy, grounded = _move_y(x, y, vy, level)
    if grounded and jump:
        # Java decides the jump in the landing tick; it moves the player next tick.
        vy = JUMP_VEL
    return (x, y, vx, vy, float(grounded))


def _move_x(x, y, vx, level):
    if vx == 0:
        return x, vx
    r0 = floor(y / TILE)
    r1 = ceil((y + PLAYER) / TILE) - 1
    if vx > 0:
        right = x + PLAYER
        for c in range(ceil(right / TILE), ceil((right + vx) / TILE)):
            for r in range(r0, r1 + 1):
                if level.tile(c, r) == SOLID:
                    return float(c * TILE - PLAYER), 0.0
    else:
        for c in range(floor(x / TILE) - 1, floor((x + vx) / TILE) - 1, -1):
            for r in range(r0, r1 + 1):
                if level.tile(c, r) == SOLID:
                    return float((c + 1) * TILE), 0.0
    return x + vx, vx


def _move_y(x, y, vy, level):
    """Returns (y, vy, grounded)."""
    c0 = floor(x / TILE)
    c1 = ceil((x + PLAYER) / TILE) - 1
    if vy > 0:
        bottom = y + PLAYER
        for r in range(ceil(bottom / TILE), ceil((bottom + vy) / TILE)):
            for c in range(c0, c1 + 1):
                if level.tile(c, r) == SOLID:
                    return float(r * TILE - PLAYER), 0.0, True
    elif vy < 0:
        for r in range(floor(y / TILE) - 1, floor((y + vy) / TILE) - 1, -1):
            for c in range(c0, c1 + 1):
                if level.tile(c, r) == SOLID:
                    return float((r + 1) * TILE), 0.0, False
    return y + vy, vy, False
