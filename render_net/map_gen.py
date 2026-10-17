"""Procedural maps for training the render net: endless variety, so it learns to READ
maps instead of memorising one.

    make_maps(B, seed=None) -> [B, 100, 100] long tensor of tile types
    make_maps_np(B, seed=None) -> the same as an int64 numpy array (no torch needed)

Half the maps are "level-like" (floors with gaps, platforms, walls, spikes on
surfaces, an exit), half are random rectangles. Sizes are log-uniform, so small
things (1-tile walls, single tiles) are as common as big blocks: thin things are
what renderers get wrong, so training must be full of them.

Anything the generator never makes, the network won't render well. Leaving
something out on purpose is fine (glitch bait), but do it deliberately.
"""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "game_py"))
from platformnet.physics import EMPTY, EXIT, GRID_H, GRID_W, SOLID, SPIKE  # noqa: E402

LEVEL_LIKE_FRACTION = 0.5


def make_maps(B, seed=None):
    import torch  # only here, so game_py/generate_data.py can use make_maps_np without torch
    return torch.from_numpy(make_maps_np(B, seed))


def make_maps_np(B, seed=None):
    rng = np.random.default_rng(seed)
    maps = np.zeros((B, GRID_H, GRID_W), dtype=np.int64)
    for b in range(B):
        if rng.random() < LEVEL_LIKE_FRACTION:
            _level_like(maps[b], rng)
        else:
            _random_rects(maps[b], rng)
    return maps


# ---- building blocks ---------------------------------------------------------

def _logsize(rng, lo, hi):
    """Log-uniform integer in [lo, hi]: as many 1-7s as 7-50s."""
    return int(min(hi, math.floor(math.exp(rng.uniform(math.log(lo), math.log(hi + 1))))))


def _stamp(m, rng, w, h, code, c=None, r=None):
    c = rng.integers(-w + 1, GRID_W) if c is None else c   # may hang off the edge
    r = rng.integers(-h + 1, GRID_H) if r is None else r
    m[max(r, 0):max(r + h, 0), max(c, 0):max(c + w, 0)] = code


def _random_rects(m, rng):
    """Pure variety: random boxes of random type, many of them thin."""
    for _ in range(rng.integers(10, 90)):
        code = rng.choice([SOLID, SPIKE, EXIT], p=[0.8, 0.15, 0.05])
        w, h = _logsize(rng, 1, 50), _logsize(rng, 1, 50)
        if rng.random() < 0.3:                                # force a line (wall or ledge)
            if rng.random() < 0.5:
                w = int(rng.integers(1, 3))
            else:
                h = int(rng.integers(1, 3))
        _stamp(m, rng, w, h, code)


def _level_like(m, rng):
    """Roughly what a platformer level looks like: something to stand on, things to jump."""
    # floor with gaps
    if rng.random() < 0.85:
        thick = _logsize(rng, 1, 15)
        m[GRID_H - thick:, :] = SOLID
        for _ in range(rng.integers(0, 5)):
            gw = _logsize(rng, 1, 15)
            gc = rng.integers(0, GRID_W)
            m[GRID_H - thick:, gc:gc + gw] = EMPTY
    # side walls / ceiling
    for side in ("left", "right", "top"):
        if rng.random() < 0.3:
            t = _logsize(rng, 1, 6)
            if side == "left":
                m[:, :t] = SOLID
            elif side == "right":
                m[:, GRID_W - t:] = SOLID
            else:
                m[:t, :] = SOLID
    # platforms: thin, from 1 tile wide up to 35
    for _ in range(rng.integers(6, 30)):
        _stamp(m, rng, _logsize(rng, 1, 35), int(rng.integers(1, 4)), SOLID)
    # walls and pillars: thin and tall
    for _ in range(rng.integers(2, 12)):
        _stamp(m, rng, int(rng.integers(1, 4)), _logsize(rng, 3, 50), SOLID)
    # chunky blocks
    for _ in range(rng.integers(1, 9)):
        _stamp(m, rng, _logsize(rng, 2, 25), _logsize(rng, 2, 25), SOLID)
    # single tiles
    for _ in range(rng.integers(0, 12)):
        m[rng.integers(0, GRID_H), rng.integers(0, GRID_W)] = SOLID
    # spikes sitting on surfaces (empty tile with solid below), in short runs
    rows, cols = np.nonzero((m[:-1] == EMPTY) & (m[1:] == SOLID))
    if len(rows):
        for i in rng.choice(len(rows), size=min(len(rows), int(rng.integers(2, 16))), replace=False):
            run = _logsize(rng, 1, 12)
            r, c = rows[i], cols[i]
            seg = slice(c, min(c + run, GRID_W))
            ok = (m[r, seg] == EMPTY) & (m[r + 1, seg] == SOLID)
            m[r, seg][ok] = SPIKE
    # an exit (sometimes none, sometimes two)
    for _ in range(rng.choice([0, 1, 1, 1, 2])):
        _stamp(m, rng, int(rng.integers(2, 8)), int(rng.integers(2, 12)), EXIT)
    # occasional stray spikes anywhere
    for _ in range(rng.integers(0, 3)):
        _stamp(m, rng, _logsize(rng, 1, 8), _logsize(rng, 1, 4), SPIKE)
