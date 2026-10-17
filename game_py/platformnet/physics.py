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
  On the ground, the surface's friction coefficient mu (FRICTION[material]) is
  your grip. It sets how fast you can change speed, both speeding up and slowing down:
      vx moves toward dir * MAX_RUN by at most mu * GROUND_ACCEL
  So releasing the keys decelerates you at mu * GROUND_ACCEL px/tick^2 until you
  stop, and a low-mu surface (e.g. ice) is slow to start, slow to stop, slow to turn.
  Standing across two materials uses the average mu of the tiles under you.
  In the air there is no grip, only air resistance proportional to speed:
      vx -= AIR_RESISTANCE * vx;  vx += dir * AIR_ACCEL
  Holding a direction balances out at AIR_ACCEL / AIR_RESISTANCE = MAX_RUN, so you
  keep running speed; letting go loses AIR_RESISTANCE (5%) of vx per tick.
"""
from math import ceil, floor

# Bump whenever constants or rules change: data recorded before is under different physics.
PHYSICS_VERSION = 6

TICK_HZ = 60
TILE = 5
PLAYER = 10
GRID_W = 100
GRID_H = 100

MAX_RUN = 3.3          # top running speed, px/tick
GROUND_ACCEL = 0.5     # max change in vx per tick on a surface with mu = 1.0
AIR_RESISTANCE = 0.05  # fraction of vx lost per tick in the air
AIR_ACCEL = AIR_RESISTANCE * MAX_RUN  # air push while holding; balances drag at MAX_RUN
GRAVITY = 0.6          # added to vy every tick
MAX_FALL = 15.0        # vy cap (collision is swept, so no tunnelling at this speed)
JUMP_VEL = -8.4        # vy set when jumping from the ground -> apex ~55 px, always the same
VX_EPS = 0.01          # |vx| below this snaps to 0, so the player fully stops

# Tile codes
EMPTY, SOLID, EXIT, SPIKE = range(4)

# Friction coefficient (grip) of each ground material. Add a material = add a tile code + an entry.
FRICTION = {SOLID: 1.0}

STATE_DIM = 4
X, Y, VX, VY = range(STATE_DIM)
STATE_NAMES = ("x", "y", "vx", "vy")


def ground_friction(x, y, level):
    """Average friction coefficient of the ground under the player, or None if in the air.
    Derived from position + tiles every tick, never stored (collisions snap to exact contact)."""
    bottom = y + PLAYER
    if bottom % TILE != 0:
        return None
    r = int(bottom // TILE)
    mus = [FRICTION[t] for c in range(floor(x / TILE), ceil((x + PLAYER) / TILE))
           if (t := level.tile(c, r)) in FRICTION]
    return sum(mus) / len(mus) if mus else None


def is_grounded(x, y, level):
    return ground_friction(x, y, level) is not None


def move_toward(v, target, delta):
    if v < target:
        return min(v + delta, target)
    if v > target:
        return max(v - delta, target)
    return v


def step(state, action, level):
    """Advance one tick. `level` needs .tile(col, row) -> tile code."""
    x, y, vx, vy = state
    input_dir, jump = action
    mu = ground_friction(x, y, level)
    grounded = mu is not None

    if grounded:
        vx = move_toward(vx, input_dir * MAX_RUN, mu * GROUND_ACCEL)
    else:
        vx -= AIR_RESISTANCE * vx
        vx += input_dir * AIR_ACCEL
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
