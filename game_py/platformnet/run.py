"""Game rules on top of step(): spikes kill (respawn at spawn), the exit finishes the level."""
from .physics import EXIT, SPIKE, step


def spawn_state(level):
    sx, sy = level.spawn
    return (sx, sy, 0.0, 0.0)


def advance(state, action, level):
    """One tick. Returns (next_state, dead, new_state, finished).

    next_state is the physics result; new_state is where the next tick starts
    (the spawn if the player died, otherwise next_state).
    """
    ns = step(state, action, level)
    if level.overlaps(ns, SPIKE):
        return ns, True, spawn_state(level), False
    return ns, False, ns, level.overlaps(ns, EXIT)
