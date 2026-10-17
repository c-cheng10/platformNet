"""ASCII room loader. One file per room, 100 columns x 100 rows, one char per 5px tile.

    .  empty     #  solid     ~  ice      ^  bounce     -  one-way
    X  spike (kills on overlap, not solid)     E  exit
    S  spawn (player's top-left tile; empty for physics)

Lines starting with ';' are metadata/comments, e.g. "; name: Ice cave".
Anything outside the grid counts as solid, so every room is sealed by the screen edge.
"""
import hashlib
from pathlib import Path

from .physics import (BOUNCE, EMPTY, EXIT, GRID_H, GRID_W, ICE, ONE_WAY, PLAYER,
                      SOLID, SPIKE, TILE)

LEVELS_DIR = Path(__file__).resolve().parent.parent / "levels"

CHAR_TO_TILE = {".": EMPTY, "#": SOLID, "~": ICE, "^": BOUNCE, "-": ONE_WAY,
                "X": SPIKE, "E": EXIT, "S": EMPTY}


class Level:
    def __init__(self, name, rows, spawn):
        self.name = name
        self.rows = rows  # GRID_H tuples of tile codes
        self.spawn = spawn  # (x, y) pixels, player's top-left

    def tile(self, c, r):
        if 0 <= c < GRID_W and 0 <= r < GRID_H:
            return self.rows[r][c]
        return SOLID

    def overlaps(self, state, code):
        """True if the player's box strictly overlaps any tile of this type."""
        x, y = state[0], state[1]
        for r in range(int(y // TILE), int(-(-(y + PLAYER) // TILE))):
            for c in range(int(x // TILE), int(-(-(x + PLAYER) // TILE))):
                if self.tile(c, r) == code:
                    return True
        return False


def parse_level(text, fallback_name="room"):
    name = fallback_name
    grid = []
    for line in text.splitlines():
        if line.startswith(";"):
            key, _, val = line[1:].partition(":")
            if key.strip() == "name":
                name = val.strip()
        elif line.strip():
            grid.append(line)
    if len(grid) != GRID_H or any(len(row) != GRID_W for row in grid):
        bad = [i for i, row in enumerate(grid) if len(row) != GRID_W]
        raise ValueError(f"{name}: need {GRID_W}x{GRID_H} grid, got {len(grid)} rows; bad rows {bad[:5]}")
    spawns = [(c, r) for r, row in enumerate(grid) for c, ch in enumerate(row) if ch == "S"]
    if len(spawns) != 1:
        raise ValueError(f"{name}: need exactly one S, found {len(spawns)}")
    rows = []
    for r, row in enumerate(grid):
        try:
            rows.append(tuple(CHAR_TO_TILE[ch] for ch in row))
        except KeyError as e:
            raise ValueError(f"{name}: unknown tile char {e} on row {r}") from None
    sc, sr = spawns[0]
    return Level(name, rows, (float(sc * TILE), float(sr * TILE)))


def load_levels(levels_dir=LEVELS_DIR):
    """Rooms in play order: zone1.txt, zone2.txt, ... Zone id = list index."""
    paths = sorted(Path(levels_dir).glob("zone*.txt"))
    if not paths:
        raise FileNotFoundError(f"no zone*.txt in {levels_dir}")
    return [parse_level(p.read_text(), p.stem) for p in paths]


def levels_hash(levels):
    """Fingerprint stored with every episode, so data from an old layout is detectable."""
    h = hashlib.sha1()
    for lv in levels:
        h.update(repr(lv.rows).encode())
    return h.hexdigest()[:12]
