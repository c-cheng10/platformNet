"""The ground-truth replay feature depends on this: replaying a recorded input
sequence through step() must reproduce the logged states bit-for-bit."""
import random

import numpy as np
import pytest

from platformnet.level import load_levels, levels_hash
from platformnet.logger import DATA_DIR, EpisodeLogger, load_episode
from platformnet.run import EV_FINISH, EV_NONE, advance, replay, spawn_state

LEVELS = load_levels()


def sticky_random_actions(n, seed):
    """Human-ish input: hold each (dir, jump) for a random number of ticks."""
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        a = (rng.choice((-1, 0, 1, 1)), rng.choice((0, 0, 1)))
        out.extend([a] * rng.randint(1, 60))
    return out[:n]


def record_episode(tmp_path, zone, actions):
    log = EpisodeLogger(tmp_path, "test", levels_hash(LEVELS))
    state = spawn_state(LEVELS[zone])
    log.begin(zone, state)
    for a in actions:
        ns, nz, nstate, event = advance(zone, state, a, LEVELS)
        log.record(zone, state, a, ns, event)
        zone, state = nz, nstate
        if event == EV_FINISH:
            break
    return load_episode(log.save())


def assert_bitwise_equal(a, b):
    assert a.dtype == b.dtype == np.float64
    assert np.array_equal(a.view(np.uint64), b.view(np.uint64))


@pytest.mark.parametrize("zone", range(len(LEVELS)))
def test_replay_is_bit_for_bit(tmp_path, zone):
    ep = record_episode(tmp_path, zone, sticky_random_actions(5000, seed=zone))
    states, next_states, zone_ids, events = replay(ep["start_zone"], ep["start_state"], ep["actions"], LEVELS)
    assert_bitwise_equal(states, ep["states"])
    assert_bitwise_equal(next_states, ep["next_states"])
    assert np.array_equal(zone_ids, ep["zone_ids"])
    assert np.array_equal(events, ep["events"])


def test_episode_format(tmp_path):
    ep = record_episode(tmp_path, 0, sticky_random_actions(800, seed=99))
    T = len(ep["actions"])
    assert ep["states"].shape == (T, 6) and ep["next_states"].shape == (T, 6)
    assert ep["actions"].shape == (T, 2) and ep["actions"].dtype == np.int8
    for k in ("episode_ids", "zone_ids", "events"):
        assert ep[k].shape == (T,)
    assert (ep["episode_ids"] == ep["episode_id"]).all()
    # Except right after a flagged teleport, next_states[t] is exactly states[t+1].
    continuous = ep["events"][:-1] == EV_NONE
    assert_bitwise_equal(ep["next_states"][:-1][continuous], ep["states"][1:][continuous])


@pytest.mark.parametrize("path", sorted(DATA_DIR.glob("**/ep_*.npz")), ids=lambda p: p.name)
def test_recorded_data_replays(path):
    """Every real episode on disk must still replay. Fails if physics changed after
    recording: re-record, or bump PHYSICS_VERSION and move the old data aside."""
    ep = load_episode(path)
    if str(ep["levels_hash"]) != levels_hash(LEVELS):
        pytest.skip("recorded on a different level layout")
    states, next_states, _, _ = replay(ep["start_zone"], ep["start_state"], ep["actions"], LEVELS)
    assert_bitwise_equal(states, ep["states"])
    assert_bitwise_equal(next_states, ep["next_states"])
