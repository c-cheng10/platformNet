"""Room sequencing on top of step(). Pure and deterministic, shared by the live game,
the replayer and (later) the headless bots, so all three agree tick for tick.

Each tick: next_state = step(...) is the physics result and is always what gets
logged as the training target. Then, if the player overlaps a spike they respawn
at the room's spawn (death); if they overlap an exit they move to the next room's
spawn; the last room's exit finishes the run. Those teleports are flagged per frame
in `events` so training never mistakes them for physics.
"""
import numpy as np

from .physics import EXIT, SPIKE, STATE_DIM, step

EV_NONE, EV_DEATH, EV_EXIT, EV_FINISH = 0, 1, 2, 3


def spawn_state(level):
    sx, sy = level.spawn
    return (sx, sy, 0.0, 0.0, 0.0, 0.0)


def advance(zone, state, action, levels):
    """One tick. Returns (next_state, new_zone, new_state, event).

    new_state is where the next tick starts; it differs from next_state only on a teleport.
    """
    level = levels[zone]
    ns = step(state, action, level)
    if level.overlaps(ns, SPIKE):
        return ns, zone, spawn_state(level), EV_DEATH
    if level.overlaps(ns, EXIT):
        if zone + 1 < len(levels):
            return ns, zone + 1, spawn_state(levels[zone + 1]), EV_EXIT
        return ns, zone, ns, EV_FINISH
    return ns, zone, ns, EV_NONE


def replay(start_zone, start_state, actions, levels):
    """Re-run a recorded input sequence. Returns states, next_states, zone_ids, events."""
    T = len(actions)
    states = np.zeros((T, STATE_DIM), dtype=np.float64)
    next_states = np.zeros((T, STATE_DIM), dtype=np.float64)
    zone_ids = np.zeros(T, dtype=np.int8)
    events = np.zeros(T, dtype=np.int8)
    zone, state = int(start_zone), tuple(float(v) for v in start_state)
    for t in range(T):
        a = (int(actions[t][0]), int(actions[t][1]))
        ns, nz, nstate, ev = advance(zone, state, a, levels)
        states[t], next_states[t], zone_ids[t], events[t] = state, ns, zone, ev
        zone, state = nz, nstate
    return states, next_states, zone_ids, events
