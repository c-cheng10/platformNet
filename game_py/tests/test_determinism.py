"""Two guarantees the project depends on:

1. Markov: every logged target (next x, y, vx, vy, dead) is a pure function of
   exactly the network's inputs for that frame (action, x, y, vx, vy, tiles).
2. Replay: re-running a recorded input sequence reproduces the log bit-for-bit
   (the ground-truth replay feature).
"""
import random

import numpy as np
import pytest

from platformnet.level import LEVEL_PATH, Level, load_level, parse_level
from platformnet.logger import DATA_DIR, EpisodeLogger, load_episode
from platformnet.physics import PHYSICS_VERSION, SPIKE, step
from platformnet.run import advance, replay, spawn_state

LEVEL = load_level()


def sticky_random_actions(n, seed):
    """Human-ish input: hold each (dir, jump) for a random number of ticks."""
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        a = (rng.choice((-1, 0, 1, 1)), rng.choice((0, 0, 1)))
        out.extend([a] * rng.randint(1, 40))
    return out[:n]


def record_episode(tmp_path, actions):
    log = EpisodeLogger(tmp_path, "test", LEVEL)
    state = spawn_state(LEVEL)
    log.begin(state)
    for a in actions:
        ns, dead, nstate, finished = advance(state, a, LEVEL)
        log.record(state, a, ns, dead)
        state = nstate
        if finished:
            break
    return load_episode(log.save())


def assert_bitwise_equal(a, b):
    assert a.dtype == b.dtype == np.float64
    assert np.array_equal(a.view(np.uint64), b.view(np.uint64))


@pytest.mark.parametrize("seed", range(6))
def test_every_frame_is_a_function_of_network_inputs_only(tmp_path, seed):
    ep = record_episode(tmp_path, sticky_random_actions(4000, seed))
    # A level rebuilt from nothing but the logged tiles (spawn is irrelevant to physics).
    lv = Level([tuple(int(v) for v in row) for row in ep["tiles"]], (0.0, 0.0))
    order = list(range(len(ep["actions"])))
    random.Random(seed).shuffle(order)  # frames in random order: no history can leak in
    for t in order:
        s = tuple(float(v) for v in ep["states"][t])
        a = (int(ep["actions"][t][0]), int(ep["actions"][t][1]))
        ns = step(s, a, lv)
        assert ns == tuple(ep["next_states"][t])
        assert int(lv.overlaps(ns, SPIKE)) == ep["dead"][t]


@pytest.mark.parametrize("seed", range(3))
def test_replay_is_bit_for_bit(tmp_path, seed):
    ep = record_episode(tmp_path, sticky_random_actions(4000, seed))
    states, next_states, dead = replay(ep["start_state"], ep["actions"], LEVEL)
    assert_bitwise_equal(states, ep["states"])
    assert_bitwise_equal(next_states, ep["next_states"])
    assert np.array_equal(dead, ep["dead"])


def test_stale_episodes_are_skipped_not_failed(tmp_path):
    ep_path = EpisodeLogger(tmp_path, "test", LEVEL)
    ep_path.begin(spawn_state(LEVEL))
    ep_path.record(spawn_state(LEVEL), (1, 0), (0.0, 0.0, 0.0, 0.0), False)  # deliberately wrong target
    path = ep_path.save()
    data = dict(load_episode(path), physics_version=np.int32(PHYSICS_VERSION - 1))
    np.savez(path, **data)
    with pytest.raises(pytest.skip.Exception):
        test_recorded_data_replays(path)


def test_level_hash_includes_spawn():
    rows = LEVEL_PATH.read_text().splitlines()
    moved = [list(r.replace("S", ".")) for r in rows]
    moved[1][1] = "S"  # same tiles, different spawn
    assert parse_level("\n".join("".join(r) for r in moved)).hash != LEVEL.hash


def test_episode_format(tmp_path):
    ep = record_episode(tmp_path, sticky_random_actions(800, seed=99))
    T = len(ep["actions"])
    assert ep["actions"].shape == (T, 2) and ep["actions"].dtype == np.int8
    assert ep["states"].shape == (T, 4) and ep["next_states"].shape == (T, 4)
    assert ep["tiles"].shape == (100, 100)
    assert ep["dead"].shape == (T,) and ep["episode_ids"].shape == (T,)
    assert int(ep["tick_hz"]) == 60
    # Except right after a death (respawn), next_states[t] is exactly states[t+1].
    alive = ep["dead"][:-1] == 0
    assert_bitwise_equal(ep["next_states"][:-1][alive], ep["states"][1:][alive])


@pytest.mark.parametrize("path", sorted(DATA_DIR.glob("**/ep_*.npz")), ids=lambda p: p.name)
def test_recorded_data_replays(path):
    """Every real episode on disk must still replay. Fails if physics changed after
    recording: re-record, or move the old data aside."""
    ep = load_episode(path)
    if int(ep.get("physics_version", -1)) != PHYSICS_VERSION:
        pytest.skip(f"recorded under physics v{int(ep.get('physics_version', -1))}, now v{PHYSICS_VERSION}")
    if str(ep.get("level_hash", "")) != LEVEL.hash:
        pytest.skip("recorded on a different level layout")
    states, next_states, _ = replay(ep["start_state"], ep["actions"], LEVEL)
    assert_bitwise_equal(states, ep["states"])
    assert_bitwise_equal(next_states, ep["next_states"])
