"""Game rules on top of step(). Pure and deterministic, shared by the live game,
the replayer and (later) the headless bots, so all three agree tick for tick.

Each tick produces exactly the network's targets:
    next_state = step(state, action, level)        -> next x, y, vx, vy
    dead       = player overlaps a spike after moving
Then the game (not the network) applies the consequences: if dead, the player
respawns at the spawn point; if they reach the exit, the run is finished.
"""
import numpy as np

from .physics import EXIT, SPIKE, STATE_DIM, step


def spawn_state(level):
    sx, sy = level.spawn
    return (sx, sy, 0.0, 0.0)


def advance(state, action, level):
    """One tick. Returns (next_state, dead, new_state, finished).

    new_state is where the next tick starts: the spawn if dead, else next_state.
    """
    ns = step(state, action, level)
    if level.overlaps(ns, SPIKE):
        return ns, True, spawn_state(level), False
    return ns, False, ns, level.overlaps(ns, EXIT)


def replay(start_state, actions, level):
    """Re-run a recorded input sequence. Returns states, next_states, dead."""
    T = len(actions)
    states = np.zeros((T, STATE_DIM), dtype=np.float64)
    next_states = np.zeros((T, STATE_DIM), dtype=np.float64)
    dead = np.zeros(T, dtype=np.uint8)
    state = tuple(float(v) for v in start_state)
    for t in range(T):
        a = (int(actions[t][0]), int(actions[t][1]))
        ns, d, nstate, _ = advance(state, a, level)
        states[t], next_states[t], dead[t] = state, ns, d
        state = nstate
    return states, next_states, dead
