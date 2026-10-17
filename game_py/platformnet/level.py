"""ASCII level loader. 100 columns x 100 rows, one char per 5px tile.

    .  empty     #  solid     X  spike (kills on overlap, not solid)
    E  exit      S  spawn (player's top-left tile; empty for physics)

Lines starting with ';' are comments. Anything outside the grid counts as solid.
"""
from pathlib import Path

from .physics import EMPTY, EXIT, GRID_H, GRID_W, PLAYER, SOLID, SPIKE, TILE

# The one levels folder for the whole repo: <repo>/levels/
LEVELS_DIR = Path(__file__).resolve().parents[2] / "levels"
LEVEL_PATH = LEVELS_DIR / "level1.txt"


def resolve_level(name=None):
    """Turn what a user typed into a level file path.

    None -> levels/level1.txt;  "level2" or "level2.txt" -> levels/level2.txt;
    anything that is already an existing path (or contains a folder) is used as-is.
    """
    if name is None:
        return LEVEL_PATH
    p = Path(name)
    if p.exists() or len(p.parts) > 1:
        return p
    return LEVELS_DIR / (p.name if p.suffix else p.name + ".txt")

CHAR_TO_TILE = {".": EMPTY, "#": SOLID, "X": SPIKE, "E": EXIT, "S": EMPTY}


class Level:
    def __init__(self, rows, spawn):
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


def parse_level(text):
    grid = [line for line in text.splitlines() if line.strip() and not line.startswith(";")]
    if len(grid) != GRID_H or any(len(row) != GRID_W for row in grid):
        bad = [i for i, row in enumerate(grid) if len(row) != GRID_W]
        raise ValueError(f"need {GRID_W}x{GRID_H} grid, got {len(grid)} rows; bad rows {bad[:5]}")
    spawns = [(c, r) for r, row in enumerate(grid) for c, ch in enumerate(row) if ch == "S"]
    if len(spawns) != 1:
        raise ValueError(f"need exactly one S, found {len(spawns)}")
    rows = []
    for r, row in enumerate(grid):
        try:
            rows.append(tuple(CHAR_TO_TILE[ch] for ch in row))
        except KeyError as e:
            raise ValueError(f"unknown tile char {e} on row {r}") from None
    sc, sr = spawns[0]
    return Level(rows, (float(sc * TILE), float(sr * TILE)))


def load_level(path=None):
    """Load a level by name ("level2") or path; default levels/level1.txt."""
    return parse_level(resolve_level(path).read_text())
