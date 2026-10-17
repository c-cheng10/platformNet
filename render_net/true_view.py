"""Ground-truth camera view: what the render net must learn to produce.

    true_view(tiles, cx, cy) -> [B, 150, 150] tile type per pixel

Plain tensor code, no neural network. Every training example comes from here,
so it must be exactly right. Runs batched on the GPU (or CPU).

The camera sees VIEW x VIEW world pixels (30 x 30 tiles) centred on (cx, cy).
For the player, the centre is (x + 5, y + 5): state stores the top-left corner of
the 10 px player. Each output pixel shows the tile under its centre point
(nearest sampling), so sub-pixel camera moves scroll the way pixel-art games do.
Anything outside the map is SOLID, the same as the game's physics (and it draws black).
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "game_py"))
from platformnet.level import load_level  # noqa: E402
from platformnet.physics import EMPTY, EXIT, GRID_H, GRID_W, SOLID, SPIKE, TILE  # noqa: E402

VIEW = 150                    # camera size in world pixels (30 tiles)
N_TYPES = 4                   # EMPTY, SOLID, EXIT, SPIKE

PALETTE = torch.tensor([      # indexed by tile type, RGB 0-255
    (0xBC, 0xDF, 0xFF),       # EMPTY: sky
    (0, 0, 0),                # SOLID
    (60, 200, 90),            # EXIT
    (220, 40, 40),            # SPIKE
], dtype=torch.uint8)
assert (EMPTY, SOLID, EXIT, SPIKE) == (0, 1, 2, 3)

# Pixel-centre offsets from the camera centre: -74.5 ... +74.5
_OFFSETS = torch.arange(VIEW, dtype=torch.float32) - VIEW / 2 + 0.5


def load_tiles(path=None):
    """A level file -> [100, 100] long tensor of tile codes (spawn counts as empty)."""
    level = load_level(path) if path else load_level()
    return torch.tensor(level.rows, dtype=torch.long)


def view_points(cx, cy):
    """World coordinates of every output pixel's centre: two [B, VIEW, VIEW] tensors (px, py)."""
    off = _OFFSETS.to(cx.device)
    px = cx[:, None, None] + off[None, None, :]   # varies along columns
    py = cy[:, None, None] + off[None, :, None]   # varies along rows
    return px.expand(-1, VIEW, VIEW), py.expand(-1, VIEW, VIEW)


def true_view(tiles, cx, cy):
    """tiles: [B, 100, 100] or [100, 100] long; cx, cy: [B] float camera centre (world px).

    Returns [B, VIEW, VIEW] long tile types (SOLID outside the map).
    """
    if tiles.dim() == 2:
        tiles = tiles.expand(cx.shape[0], -1, -1)
    B = cx.shape[0]
    px, py = view_points(cx.float(), cy.float())
    col = torch.floor(px / TILE).long()
    row = torch.floor(py / TILE).long()
    inside = (col >= 0) & (col < GRID_W) & (row >= 0) & (row < GRID_H)
    flat = tiles.reshape(B, -1)
    idx = (row.clamp(0, GRID_H - 1) * GRID_W + col.clamp(0, GRID_W - 1)).reshape(B, -1)
    out = torch.gather(flat, 1, idx).reshape(B, VIEW, VIEW)
    return torch.where(inside, out, torch.full_like(out, SOLID))


def to_rgb(types):
    """[..., H, W] tile types -> [..., H, W, 3] uint8 RGB image."""
    return PALETTE.to(types.device)[types]
