"""Game rules on top of step(). Pure and deterministic, shared by the live game,
the replayer and (later) the headless bots, so all three agree tick for tick.

Each tick, next_state = step(...) is the physics result and is always what gets
logged as the training target. Then, if the player overlaps a spike they respawn
(death); if they overlap the exit the run is finished. The respawn teleport is
flagged in `events` so training never mistakes it for physics.
"""
import numpy as np

from .physics import EXIT, SPIKE, STATE_DIM, step

EV_NONE, EV_DEATH, EV_FINISH = 0, 1, 2


def spawn_state(level):
    sx, sy = level.spawn
    return (sx, sy, 0.0, 0.0, 0.0)


def advance(state, action, level):
    """One tick. Returns (next_state, new_state, event).

    new_state is where the next tick starts; it differs from next_state only on death.
    """
    ns = step(state, action, level)
    if level.overlaps(ns, SPIKE):
        return ns, spawn_state(level), EV_DEATH
    if level.overlaps(ns, EXIT):
        return ns, ns, EV_FINISH
    return ns, ns, EV_NONE


def replay(start_state, actions, level):
    """Re-run a recorded input sequence. Returns states, next_states, events."""
    T = len(actions)
    states = np.zeros((T, STATE_DIM), dtype=np.float64)
    next_states = np.zeros((T, STATE_DIM), dtype=np.float64)
    events = np.zeros(T, dtype=np.int8)
    state = tuple(float(v) for v in start_state)
    for t in range(T):
        a = (int(actions[t][0]), int(actions[t][1]))
        ns, nstate, ev = advance(state, a, level)
        states[t], next_states[t], events[t] = state, ns, ev
        state = nstate
    return states, next_states, events
