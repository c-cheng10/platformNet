"""Episode logging. One .npz per episode, laid out as the network's inputs/targets:

  inputs
    actions      [T,2]     int8     (input_dir, jump) pressed at tick t  <- also the replay input
    states       [T,4]     float64  x, y, vx, vy at tick t
    tiles        [100,100] int8     the level grid (0 empty, 1 solid, 2 exit, 3 spike), same for every tick
  targets
    next_states  [T,4]     float64  x, y, vx, vy after the tick
    dead         [T]       uint8    1 if the tick ended on a spike (the game then respawns the player)
  bookkeeping
    episode_ids [T], start_state, episode_id, source, finished, physics_version,
    tick_hz, level_hash, recorded_at

float64 is deliberate: step() runs on Python floats, so replay is bit-for-bit.
Cast to float32 at training time, not here.
"""
import re
import time
from pathlib import Path

import numpy as np

from .physics import PHYSICS_VERSION, STATE_DIM, TICK_HZ

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def next_episode_id(out_dir):
    ids = [int(m.group(1)) for p in Path(out_dir).glob("ep_*.npz")
           if (m := re.match(r"ep_(\d+)", p.stem))]
    return max(ids, default=-1) + 1


class EpisodeLogger:
    def __init__(self, out_dir, source, level):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.source = source
        self.level = level
        self.begin(None)

    def begin(self, state):
        self.states, self.actions, self.next_states, self.dead = [], [], [], []
        self.start_state = state

    def record(self, state, action, next_state, dead):
        self.states.append(state)
        self.actions.append(action)
        self.next_states.append(next_state)
        self.dead.append(dead)

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
            actions=np.asarray(self.actions, dtype=np.int8).reshape(T, 2),
            states=np.asarray(self.states, dtype=np.float64).reshape(T, STATE_DIM),
            tiles=np.asarray(self.level.rows, dtype=np.int8),
            next_states=np.asarray(self.next_states, dtype=np.float64).reshape(T, STATE_DIM),
            dead=np.asarray(self.dead, dtype=np.uint8),
            episode_ids=np.full(T, ep, dtype=np.int32),
            start_state=np.asarray(self.start_state, dtype=np.float64),
            episode_id=np.int32(ep),
            source=np.str_(self.source),
            finished=np.bool_(finished),
            physics_version=np.int32(PHYSICS_VERSION),
            tick_hz=np.int32(TICK_HZ),
            level_hash=np.str_(self.level.hash),
            recorded_at=np.str_(time.strftime("%Y-%m-%dT%H:%M:%S")),
        )
        self.begin(self.start_state)
        return path


def load_episode(path):
    with np.load(path) as z:
        return {k: z[k] for k in z.files}
