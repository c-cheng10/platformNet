"""true_view must be exactly right: the network learns whatever it outputs."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from true_view import N_TYPES, VIEW, load_tiles, true_view  # noqa: E402
from platformnet.physics import EMPTY, SOLID, TILE  # noqa: E402

T = lambda *v: torch.tensor(v, dtype=torch.float32)  # noqa: E731


def single_tile_map(c, r, code=SOLID):
    tiles = torch.zeros(100, 100, dtype=torch.long)
    tiles[r, c] = code
    return tiles


def test_shape_and_dtype():
    out = true_view(load_tiles(), T(250, 100), T(250, 400))
    assert out.shape == (2, VIEW, VIEW) and out.dtype == torch.long


def test_tile_lands_where_expected():
    # Tile (50, 50) covers world px 250..255. Camera centred at 250 -> view pixel 75 is world 250.5.
    tiles = single_tile_map(50, 50)
    v = true_view(tiles, T(250.0), T(250.0))[0]
    assert v[75, 75] == SOLID and v[79, 79] == SOLID
    assert v[74, 75] == EMPTY and v[80, 75] == EMPTY and v[75, 74] == EMPTY
    assert (v == SOLID).sum() == TILE * TILE


def test_rows_are_y_and_columns_are_x():
    tiles = single_tile_map(52, 50)          # 2 tiles to the right of the centre tile
    v = true_view(tiles, T(250.0), T(250.0))[0]
    assert v[75, 85] == SOLID and v[85, 75] == EMPTY


def test_off_map_is_solid():
    empty = torch.zeros(100, 100, dtype=torch.long)
    v = true_view(empty, T(0.0), T(0.0))[0]   # camera at the map's top-left corner
    assert (v[:75, :] == SOLID).all() and (v[:, :75] == SOLID).all()
    assert (v[75:, 75:] == EMPTY).all()
    assert int(v.max()) < N_TYPES


def test_subpixel_scrolling_moves_edges_one_pixel_at_a_time():
    tiles = single_tile_map(50, 50)
    cols = []
    for k in range(11):                      # camera moves 0.1 px at a time over 1 px
        v = true_view(tiles, T(250.0 + k * 0.1), T(250.0))[0]
        cols.append(int((v[77] == SOLID).nonzero()[0]))
    assert cols[0] == 75 and cols[-1] == 74  # the tile slid exactly one pixel left
    assert all(b - a in (0, -1) for a, b in zip(cols, cols[1:]))


def test_batch_matches_one_at_a_time():
    tiles = load_tiles()
    cx, cy = torch.rand(16) * 600 - 50, torch.rand(16) * 600 - 50
    batch = true_view(tiles, cx, cy)
    for i in range(16):
        assert torch.equal(batch[i], true_view(tiles, cx[i:i + 1], cy[i:i + 1])[0])


def test_matches_the_game_level_tile_lookup():
    from platformnet.level import load_level
    level, tiles = load_level(), load_tiles()
    cx, cy = T(123.4, 377.7, 260.0), T(301.2, 99.9, 440.0)
    v = true_view(tiles, cx, cy)
    for b in range(3):
        for i, j in [(0, 0), (75, 75), (149, 3), (20, 140)]:
            wx, wy = cx[b] - VIEW / 2 + 0.5 + j, cy[b] - VIEW / 2 + 0.5 + i
            assert v[b, i, j] == level.tile(int(wx // TILE), int(wy // TILE))  # off-map: both SOLID
