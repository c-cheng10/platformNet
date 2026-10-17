"""Sketch the level as rectangles and write levels/level1.txt.

levels/level1.txt is the source of truth. Re-running this overwrites hand edits.
Coordinates are tile (col, row), inclusive, 100x100 grid of 5px tiles, row 0 at top.
Physics reference (60 ticks/s): jump apex ~78 px (~16 tiles), top speed 3.3 px/tick, jump covers ~80 px
horizontally at full speed, player 2x2 tiles.
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
rect(36, 90, 43, 94, ".")     # pit (40 px wide)...
rect(36, 94, 43, 94, "X")     # ...lined with spikes
rect(54, 82, 64, 83, "#")     # floating platform (40 px up)
rect(72, 82, 72, 89, "#")     # thin wall (40 px tall, 5 px thick)
rect(76, 89, 79, 89, "X")     # spikes right behind the wall
rect(93, 80, 98, 89, "E")     # exit
rect(3, 88, 3, 88, "S")       # spawn

if __name__ == "__main__":
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text("\n".join("".join(row) for row in g) + "\n")
    print(f"wrote {OUT.name}")
