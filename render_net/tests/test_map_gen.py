import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from map_gen import make_maps  # noqa: E402
from true_view import EMPTY, EXIT, N_TYPES, SOLID, SPIKE, true_view  # noqa: E402


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
    thin_wall = m[:, :, 1:-1] & ~m[:, :, :-2] & ~m[:, :, 2:]     # solid with empty left and right
    thin_ledge = m[:, 1:-1, :] & ~m[:, :-2, :] & ~m[:, 2:, :]    # solid with empty above and below
    per_map = (thin_wall.flatten(1).any(1) & thin_ledge.flatten(1).any(1)).float().mean()
    assert per_map > 0.9


def test_maps_work_with_true_view():
    m = make_maps(4, seed=4)
    v = true_view(m, torch.tensor([10.0, 250.0, 499.0, 77.7]), torch.tensor([5.0, 250.0, 300.0, 480.0]))
    assert v.shape == (4, 150, 150)
