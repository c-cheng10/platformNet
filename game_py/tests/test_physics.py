"""Checks for the Java-port fixes, on tiny hand-made levels."""
from platformnet.level import parse_level
from platformnet.physics import JUMP_VEL, MAX_FALL, PLAYER, step
from platformnet.run import EV_DEATH, advance


def level(solid=(), spike=()):
    """100x100 level from inclusive (c0, r0, c1, r1) boxes. Spawn at tile (1, 1)."""
    g = [["."] * 100 for _ in range(100)]
    for ch, boxes in (("#", solid), ("X", spike)):
        for c0, r0, c1, r1 in boxes:
            for r in range(r0, r1 + 1):
                for c in range(c0, c1 + 1):
                    g[r][c] = ch
    g[1][1] = "S"
    return parse_level("\n".join("".join(r) for r in g))


FLOOR = (0, 90, 99, 99)  # floor top at y=450


def run(state, action, lv, n):
    for _ in range(n):
        state = step(state, action, lv)
    return state


def test_landing_snaps_exactly_regardless_of_drop_height():
    lv = level(solid=[FLOOR])
    for y0 in (430.0, 300.0, 50.0, 123.456):
        s = run((200.0, y0, 0, 0, 0), (0, 0), lv, 300)
        assert s[1] == 450 - PLAYER and s[3] == 0 and s[4] == 1.0


def test_wall_contact_is_exact():
    lv = level(solid=[FLOOR, (60, 0, 60, 89)])  # wall face at x=300
    for x0 in (100.0, 250.0, 283.7):
        s = run((x0, 440.0, 0, 0, 1), (1, 0), lv, 300)
        assert s[0] == 300 - PLAYER and s[2] == 0


def test_fall_speed_is_capped():
    s = run((200.0, 0.0, 0, 0, 0), (0, 0), level(), 25)
    assert s[3] <= MAX_FALL


def test_no_tunnelling_through_thin_floor_at_max_fall():
    lv = level(solid=[(30, 60, 60, 60)])  # 5px ledge at y=300
    s = run((200.0, 3.0, 0.0, MAX_FALL, 0), (0, 0), lv, 200)
    assert s[1] == 300 - PLAYER and s[4] == 1.0


def test_no_tunnelling_through_thin_wall_at_top_speed():
    lv = level(solid=[FLOOR, (60, 0, 60, 89)])
    s = run((100.0, 440.0, 0, 0, 1), (1, 0), lv, 500)
    assert s[0] + PLAYER <= 300


def test_jump_is_fixed_height_and_matches_java():
    lv = level(solid=[FLOOR])
    apexes = []
    for hold in (1, 5, 200):  # releasing early must not change the height
        s = step((200.0, 440.0, 0, 0, 1), (0, 1), lv)
        assert s[3] == JUMP_VEL
        top = 440.0
        for t in range(80):
            s = step(s, (0, 1 if t < hold and s[4] == 0 else 0), lv)
            top = min(top, s[1])
        apexes.append(440 - top)
    assert apexes[0] == apexes[1] == apexes[2] and 95 < apexes[0] < 110  # ~103 px


def test_friction_stops_the_player_cleanly():
    s = run((10.0, 440.0, 0, 0, 1), (1, 0), level(solid=[FLOOR]), 100)
    s = run(s, (0, 0), level(solid=[FLOOR]), 100)
    assert s[2] == 0.0


def test_spike_kills_and_flags_respawn():
    lv = level(solid=[FLOOR], spike=[(30, 89, 60, 89)])
    s = (200.0, 400.0, 0, 0, 0)
    for _ in range(100):
        ns, s, ev = advance(s, (0, 0), lv)
        if ev == EV_DEATH:
            assert s[:2] == lv.spawn and ns[:2] != lv.spawn
            return
    raise AssertionError("never died")
