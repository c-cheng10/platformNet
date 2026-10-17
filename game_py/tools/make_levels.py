"""Sketch the level as rectangles and write levels/level1.txt.

levels/level1.txt is the source of truth. Re-running this overwrites hand edits.
Coordinates are tile (col, row), inclusive, 100x100 grid of 5px tiles, row 0 at top.
Physics reference: jump apex ~103 px (~20 tiles), top speed ~4 px/tick, player 2x2 tiles.
"""
from pathlib import Path

W = H = 100
OUT = Path(__file__).resolve().parent.parent / "levels" / "level1.txt"

g = [["."] * W for _ in range(H)]


def rect(c0, r0, c1, r1, ch):
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            g[r][c] = ch


rect(0, 90, 99, 99, "#")      # floor
rect(20, 84, 27, 89, "#")     # low step (30 px)
rect(36, 90, 47, 94, ".")     # pit...
rect(36, 94, 47, 94, "X")     # ...lined with spikes
rect(54, 76, 64, 77, "#")     # floating platform (70 px up)
rect(72, 72, 72, 89, "#")     # thin wall (90 px tall, 5 px thick)
rect(76, 89, 81, 89, "X")     # spikes right behind the wall
rect(93, 80, 98, 89, "E")     # exit
rect(3, 88, 3, 88, "S")       # spawn

if __name__ == "__main__":
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text("\n".join("".join(row) for row in g) + "\n")
    print(f"wrote {OUT.name}")
