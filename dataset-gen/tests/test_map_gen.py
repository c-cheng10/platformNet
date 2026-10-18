import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from map_gen import make_maps
from platformnet.physics import EMPTY, EXIT, SOLID, SPIKE

N_TYPES = 4


def test_shape_dtype_and_valid_codes():
    m = make_maps(32, seed=1)
    assert m.shape == (32, 100, 100) and m.dtype == torch.long
    assert int(m.min()) >= 0 and int(m.max()) < N_TYPES


def test_same_seed_same_maps_different_seed_different_maps():
    assert torch.equal(make_maps(8, seed=5), make_maps(8, seed=5))
    assert not torch.equal(make_maps(8, seed=5), make_maps(8, seed=6))


def test_every_tile_type_shows_up():
    m = make_maps(200, seed=2)
    for code in (EMPTY, SOLID, EXIT, SPIKE):
        assert (m == code).any()


def test_thin_structures_are_common():
    """1-tile-thick walls/ledges are the hard case, so they must be everywhere in training."""
    m = make_maps(100, seed=3) == SOLID
    thin_wall = m[:, :, 1:-1] & ~m[:, :, :-2] & ~m[:, :, 2:]
    thin_ledge = m[:, 1:-1, :] & ~m[:, :-2, :] & ~m[:, 2:, :]
    per_map = (thin_wall.flatten(1).any(1) & thin_ledge.flatten(1).any(1)).float().mean()
    assert per_map > 0.9
