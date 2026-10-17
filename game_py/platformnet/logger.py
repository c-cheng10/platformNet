"""Episode logging. One .npz per episode:

    states       [T,6] float64   state at tick t  (x, y, vx, vy, grounded, on_ice)
    actions      [T,2] int8      (input_dir, jump) at tick t  <- the raw input sequence
    next_states  [T,6] float64   physics result of (states[t], actions[t])
    episode_ids  [T]   int32     per-frame episode id
    zone_ids     [T]   int8      room the tick happened in (the model needs this:
                                 all rooms share the same pixel coordinates)
    events       [T]   int8      0 none, 1 death, 2 exit, 3 finish (teleport after this tick)
    start_zone, start_state      enough to replay the run exactly from `actions`
    episode_id, source, finished, physics_version, levels_hash, recorded_at

float64 is deliberate: step() runs on Python floats, so replay is bit-for-bit.
Cast to float32 at training time, not here.
"""
import re
import time
from pathlib import Path

import numpy as np

from .physics import PHYSICS_VERSION, STATE_DIM

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def next_episode_id(out_dir):
    ids = [int(m.group(1)) for p in Path(out_dir).glob("ep_*.npz")
           if (m := re.match(r"ep_(\d+)", p.stem))]
    return max(ids, default=-1) + 1


class EpisodeLogger:
    def __init__(self, out_dir, source, levels_hash):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.source = source
        self.levels_hash = levels_hash
        self.begin(None, None)

    def begin(self, zone, state):
        self.states, self.actions, self.next_states, self.zone_ids, self.events = [], [], [], [], []
        self.start_zone = zone
        self.start_state = state

    def record(self, zone, state, action, next_state, event):
        self.states.append(state)
        self.actions.append(action)
        self.next_states.append(next_state)
        self.zone_ids.append(zone)
        self.events.append(event)

    def __len__(self):
        return len(self.actions)

    def save(self, finished=False):
        """Write the episode if it has any ticks. Returns the path or None."""
        if not self.actions:
            return None
        ep = next_episode_id(self.out_dir)
        T = len(self.actions)
        path = self.out_dir / f"ep_{ep:06d}.npz"
        np.savez_compressed(
            path,
            states=np.asarray(self.states, dtype=np.float64).reshape(T, STATE_DIM),
            actions=np.asarray(self.actions, dtype=np.int8).reshape(T, 2),
            next_states=np.asarray(self.next_states, dtype=np.float64).reshape(T, STATE_DIM),
            episode_ids=np.full(T, ep, dtype=np.int32),
            zone_ids=np.asarray(self.zone_ids, dtype=np.int8),
            events=np.asarray(self.events, dtype=np.int8),
            start_zone=np.int8(self.start_zone),
            start_state=np.asarray(self.start_state, dtype=np.float64),
            episode_id=np.int32(ep),
            source=np.str_(self.source),
            finished=np.bool_(finished),
            physics_version=np.int32(PHYSICS_VERSION),
            levels_hash=np.str_(self.levels_hash),
            recorded_at=np.str_(time.strftime("%Y-%m-%dT%H:%M:%S")),
        )
        self.begin(self.start_zone, self.start_state)
        return path


def load_episode(path):
    with np.load(path) as z:
        return {k: z[k] for k in z.files}
