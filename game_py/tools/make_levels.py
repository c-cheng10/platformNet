"""Sketch the six rooms as rectangles and write levels/zone*.txt.

The .txt files are the source of truth. Once someone hand-edits them, re-running
this overwrites their changes, so only re-run it for a deliberate re-sketch.
Coordinates are tile (col, row), inclusive, 100x100 grid of 5px tiles, row 0 at top.
Physics reference (Java constants): jump apex ~103 px (~20 tiles), bounce apex
~213 px (~42 tiles), top run speed ~4 px/tick, player 2x2 tiles.
"""
from pathlib import Path

W = H = 100
OUT = Path(__file__).resolve().parent.parent / "levels"


class Room:
    def __init__(self, name):
        self.name = name
        self.g = [["."] * W for _ in range(H)]

    def rect(self, c0, r0, c1, r1, ch):
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                self.g[r][c] = ch
        return self

    def text(self):
        return f"; name: {self.name}\n" + "\n".join("".join(row) for row in self.g) + "\n"


def tutorial():
    m = Room("Tutorial strip")
    m.rect(0, 90, 99, 99, "#")             # floor
    m.rect(25, 84, 32, 89, "#")            # low step (30 px)
    m.rect(44, 90, 55, 94, ".")            # ditch...
    m.rect(44, 94, 55, 94, "X")            # ...with spikes in it
    m.rect(62, 76, 72, 77, "#")            # floating platform (70 px up)
    m.rect(80, 75, 85, 89, "#")            # tall block (75 px)
    m.rect(93, 80, 98, 89, "E")
    m.rect(4, 88, 4, 88, "S")
    return m


def ice_cave():
    m = Room("Ice cave")
    m.rect(0, 0, 99, 39, "#")              # cave ceiling
    m.rect(0, 60, 12, 99, "#")             # spawn ledge (normal ground)
    for c0, c1, top in [(13, 27, 66), (28, 42, 72), (43, 55, 85), (64, 76, 85), (83, 99, 85)]:
        m.rect(c0, top, c1, 99, "#")
        m.rect(c0, top, c1, top + 1, "~")  # ice skin on every step / floor
    m.rect(56, 96, 63, 99, "#")            # pits with spikes
    m.rect(56, 95, 63, 95, "X")
    m.rect(77, 96, 82, 99, "#")
    m.rect(77, 95, 82, 95, "X")
    m.rect(68, 40, 71, 72, "#")            # stalactite: bonk if you jump under it
    m.rect(95, 77, 98, 84, "E")
    m.rect(4, 58, 4, 58, "S")
    return m


def bounce_tower():
    m = Room("Bounce tower")
    m.rect(0, 95, 99, 99, "#")             # floor
    m.rect(0, 0, 9, 99, "#")               # left wall
    m.rect(18, 95, 28, 95, "^")            # pad flush with floor
    m.rect(36, 58, 60, 60, "#")            # ledge A (top 290 px; bounce clears by ~28 px)
    m.rect(48, 58, 60, 58, "^")            # pad on ledge A
    m.rect(70, 24, 99, 26, "#")            # ledge B (top 120 px)
    m.rect(92, 14, 97, 23, "E")
    m.rect(12, 93, 12, 93, "S")
    return m


def one_way_shaft():
    m = Room("One-way shaft")
    m.rect(0, 0, 29, 99, "#")
    m.rect(71, 0, 99, 99, "#")
    m.rect(30, 95, 70, 99, "#")
    for i, row in enumerate([80, 64, 48, 32, 16]):  # 80 px apart, jump is ~103
        c0, c1 = (30, 48) if i % 2 == 0 else (52, 70)
        m.rect(c0, row, c1, row, "-")
    m.rect(33, 6, 38, 15, "E")
    m.rect(48, 93, 48, 93, "S")
    return m


def gauntlet():
    m = Room("Combo gauntlet")
    m.rect(0, 90, 99, 99, "#")
    m.rect(0, 90, 40, 91, "~")             # ice runway
    m.rect(41, 90, 52, 90, "^")            # pad 1
    m.rect(57, 54, 57, 89, "#")            # THIN wall 1 (1 tile = 5px), top 270 px
    m.rect(63, 90, 72, 90, "^")            # pad 2
    m.rect(78, 51, 78, 89, "#")            # THIN wall 2, top 255 px (bounce clears by ~18)
    m.rect(79, 90, 99, 91, "~")
    m.rect(94, 80, 98, 89, "E")
    m.rect(3, 88, 3, 88, "S")
    return m


def thick_wall():
    m = Room("Thick wall")
    m.rect(0, 90, 99, 99, "#")
    m.rect(60, 15, 79, 99, "#")            # THICK wall, 100px; route goes over the top
    for c0, r, c1 in [(10, 76, 22), (28, 62, 40), (12, 48, 24), (30, 34, 42), (46, 20, 58)]:
        m.rect(c0, r, c1, r + 1, "#")      # staircase up the left side, 70 px steps
    m.rect(90, 80, 95, 89, "E")            # flag behind the wall
    m.rect(3, 88, 3, 88, "S")
    return m


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    for i, room in enumerate([tutorial(), ice_cave(), bounce_tower(), one_way_shaft(), gauntlet(), thick_wall()], 1):
        (OUT / f"zone{i}.txt").write_text(room.text())
        print(f"zone{i}.txt  {room.name}")
