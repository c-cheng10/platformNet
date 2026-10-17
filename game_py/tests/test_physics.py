"""Checks for the Java-port fixes, on tiny hand-made rooms."""
from platformnet.level import parse_level
from platformnet.physics import (BOUNCE_VEL, GRAVITY, JUMP_VEL, MAX_FALL, PLAYER,
                                 TILE, step)
from platformnet.run import EV_DEATH, advance


def room(**rects):
    """100x100 room; rects maps a char to [(c0, r0, c1, r1), ...] inclusive. Spawn at (1, 1)."""
    g = [["."] * 100 for _ in range(100)]
    for ch, boxes in rects.items():
        ch = {"solid": "#", "ice": "~", "bounce": "^", "oneway": "-", "spike": "X"}[ch]
        for c0, r0, c1, r1 in boxes:
            for r in range(r0, r1 + 1):
                for c in range(c0, c1 + 1):
                    g[r][c] = ch
    g[1][1] = "S"
    return parse_level("\n".join("".join(r) for r in g))


FLOOR = (0, 90, 99, 99)  # floor top at y=450


def run(state, action, level, n):
    for _ in range(n):
        state = step(state, action, level)
    return state


def test_landing_snaps_exactly_regardless_of_drop_height():
    lv = room(solid=[FLOOR])
    for y0 in (430.0, 300.0, 50.0, 123.456):
        s = run((200.0, y0, 0, 0, 0, 0), (0, 0), lv, 300)
        assert s[1] == 450 - PLAYER and s[3] == 0 and s[4] == 1.0


def test_wall_contact_is_exact():
    lv = room(solid=[FLOOR, (60, 0, 60, 89)])  # wall face at x=300
    for x0 in (100.0, 250.0, 283.7):
        s = run((x0, 440.0, 0, 0, 1, 0), (1, 0), lv, 300)
        assert s[0] == 300 - PLAYER and s[2] == 0


def test_fall_speed_is_capped():
    lv = room()
    s = run((200.0, 0.0, 0, 0, 0, 0), (0, 0), lv, 25)
    assert s[3] <= MAX_FALL


def test_no_tunnelling_through_thin_one_way_at_max_fall():
    lv = room(solid=[FLOOR], oneway=[(30, 60, 60, 60)])  # 5px platform at y=300
    s = (200.0, 3.0, 0.0, MAX_FALL, 0, 0)
    for _ in range(200):
        s = step(s, (0, 0), lv)
    assert s[1] == 300 - PLAYER and s[4] == 1.0


def test_no_tunnelling_through_thin_wall_at_top_speed():
    lv = room(solid=[FLOOR, (60, 0, 60, 89)])
    s = run((100.0, 440.0, 0, 0, 1, 0), (1, 0), lv, 500)
    assert s[0] + PLAYER <= 300


def test_one_way_passes_from_below_and_holds_from_above():
    lv = room(solid=[FLOOR], oneway=[(30, 80, 60, 80)])  # 30 px above the floor
    s = (200.0, 440.0, 0, 0, 1, 0)
    s = step(s, (0, 1), lv)                      # jump (held) from the floor
    peak = s
    for _ in range(60):
        s = step(s, (0, 0), lv)
        peak = min(peak, s, key=lambda q: q[1])
    assert peak[1] < 400 - PLAYER                # went up through the platform
    assert s[1] == 400 - PLAYER and s[4] == 1.0  # and landed on top of it


def test_jump_apex_matches_java():
    lv = room(solid=[FLOOR])
    s = step((200.0, 440.0, 0, 0, 1, 0), (0, 1), lv)
    assert s[3] == JUMP_VEL
    top = 440.0
    for _ in range(80):
        s = step(s, (0, 0), lv)
        top = min(top, s[1])
    assert 95 < 440 - top < 110                  # ~103 px, same as Game.java


def test_ice_slides_further_than_ground():
    def slide(tile):
        lv = room(**{tile: [FLOOR]})
        s = run((10.0, 440.0, 0, 0, 1, 0), (1, 0), lv, 100)
        x0 = s[0]
        s = run(s, (0, 0), lv, 600)
        assert s[2] == 0
        return s[0] - x0
    assert slide("ice") > 4 * slide("solid")     # ~106 px vs ~22 px at ICE_FRICTION 0.97


def test_bounce_pad_launches():
    lv = room(bounce=[FLOOR])
    s = (200.0, 400.0, 0, 0, 0, 0)
    while s[3] >= 0:                             # fall until the pad launches us
        s = step(s, (0, 0), lv)
    assert s[1] == 450 - PLAYER and s[3] == -BOUNCE_VEL and s[4] == 0.0
    top = s[1]
    for _ in range(80):
        s = step(s, (0, 0), lv)
        top = min(top, s[1])
    assert 440 - top > 2 * 95                    # roughly twice a jump


def test_spike_kills_and_flags_teleport():
    lv = room(solid=[FLOOR], spike=[(30, 89, 60, 89)])
    zone, s = 0, (200.0, 400.0, 0, 0, 0, 0)
    for _ in range(100):
        ns, zone, s, ev = advance(zone, s, (0, 0), [lv])
        if ev == EV_DEATH:
            assert s[:2] == lv.spawn and ns[:2] != lv.spawn
            return
    raise AssertionError("never died")
