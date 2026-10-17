"""Ground-truth physics. Pure Python, no Pygame, so bots can run it headless:

    step(state, action, level) -> next_state

THE RULE (the world model depends on it): the next frame is a function of exactly
what the network sees, and nothing else:

    inputs:  action (input_dir, jump), state (x, y, vx, vy), every tile in the level
    outputs: next (x, y, vx, vy) and dead

So there is no hidden state. "grounded" is not stored; it is recomputed each tick
from position + tiles (collisions snap to exact contact, so standing on a floor is
an exact, checkable fact). No timers, no edge-triggered input (holding jump just
re-jumps on landing), no randomness. If you add a rule that needs anything else,
it must become a network input too.

Units: pixels and pixels/tick at a fixed 60 ticks/s. (x, y) is the player's
top-left, y points down. The level is 100x100 tiles of 5 px (500x500 screen).

Horizontal movement, per tick:
    holding a direction:     vx = (vx + dir * ACCEL) * MOVE_DAMP   -> top speed ~3.3 px/tick
    no input, on the ground: vx *= GROUND_FRICTION                  -> stops in a few ticks
    no input, in the air:    vx *= AIR_DRAG                         -> momentum carries, slowly bleeds off
"""
from math import ceil, floor

# Bump whenever constants or rules change: data recorded before is under different physics.
PHYSICS_VERSION = 4

TICK_HZ = 60
TILE = 5
PLAYER = 10
GRID_W = 100
GRID_H = 100

ACCEL = 1.1            # added per tick while a direction is held
MOVE_DAMP = 0.75       # multiplier while a direction is held; top speed = ACCEL*MOVE_DAMP/(1-MOVE_DAMP) = 3.3
GROUND_FRICTION = 0.6  # multiplier on the ground with no input (quick stop)
AIR_DRAG = 0.95        # multiplier in the air with no input (keeps most momentum)
GRAVITY = 0.85         # added to vy every tick
MAX_FALL = 15.0        # vy cap (collision is swept, so no tunnelling at this speed)
JUMP_VEL = -10.0       # vy set when jumping from the ground -> apex ~55 px, always the same
VX_EPS = 0.01          # |vx| below this snaps to 0, so the player fully stops

# Tile codes
EMPTY, SOLID, EXIT, SPIKE = range(4)

STATE_DIM = 4
X, Y, VX, VY = range(STATE_DIM)
STATE_NAMES = ("x", "y", "vx", "vy")


def is_grounded(x, y, level):
    """Standing exactly on a solid tile. Derived from position + tiles, never stored."""
    bottom = y + PLAYER
    if bottom % TILE != 0:
        return False
    r = int(bottom // TILE)
    return any(level.tile(c, r) == SOLID
               for c in range(floor(x / TILE), ceil((x + PLAYER) / TILE)))


def step(state, action, level):
    """Advance one tick. `level` needs .tile(col, row) -> tile code."""
    x, y, vx, vy = state
    input_dir, jump = action
    grounded = is_grounded(x, y, level)

    if input_dir != 0:
        vx = (vx + input_dir * ACCEL) * MOVE_DAMP
    else:
        vx *= GROUND_FRICTION if grounded else AIR_DRAG
    if -VX_EPS < vx < VX_EPS:
        vx = 0.0

    if grounded and jump:
        vy = JUMP_VEL
    vy = min(vy + GRAVITY, MAX_FALL)

    x, vx = _move_x(x, y, vx, level)
    y, vy = _move_y(x, y, vy, level)
    return (x, y, vx, vy)


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
    c0 = floor(x / TILE)
    c1 = ceil((x + PLAYER) / TILE) - 1
    if vy > 0:
        bottom = y + PLAYER
        for r in range(ceil(bottom / TILE), ceil((bottom + vy) / TILE)):
            for c in range(c0, c1 + 1):
                if level.tile(c, r) == SOLID:
                    return float(r * TILE - PLAYER), 0.0
    elif vy < 0:
        for r in range(floor(y / TILE) - 1, floor((y + vy) / TILE) - 1, -1):
            for c in range(c0, c1 + 1):
                if level.tile(c, r) == SOLID:
                    return float((r + 1) * TILE), 0.0
    return y + vy, vy
