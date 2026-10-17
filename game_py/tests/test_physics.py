"""Behaviour checks on tiny hand-made levels."""
from platformnet.level import parse_level
from platformnet.physics import (ACCEL, AIR_DRAG, JUMP_VEL, MAX_FALL, MOVE_DAMP,
                                 PLAYER, is_grounded, step)
from platformnet.run import advance


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
GROUND = 450.0 - PLAYER  # y of a player standing on FLOOR


def run(state, action, lv, n):
    for _ in range(n):
        state = step(state, action, lv)
    return state


def test_landing_snaps_exactly_and_grounded_is_derivable():
    lv = level(solid=[FLOOR])
    for y0 in (430.0, 300.0, 50.0, 123.456):
        s = run((200.0, y0, 0, 0), (0, 0), lv, 200)
        assert s == (200.0, GROUND, 0.0, 0.0)
        assert is_grounded(s[0], s[1], lv)
    assert not is_grounded(200.0, GROUND - 0.5, lv)


def test_wall_contact_is_exact():
    lv = level(solid=[FLOOR, (60, 0, 60, 89)])  # wall face at x=300
    for x0 in (100.0, 250.0, 283.7):
        s = run((x0, GROUND, 0, 0), (1, 0), lv, 300)
        assert s[0] == 300 - PLAYER and s[2] == 0


def test_top_speed():
    s = run((0.0, GROUND, 0, 0), (1, 0), level(solid=[FLOOR]), 60)
    assert abs(s[2] - ACCEL * MOVE_DAMP / (1 - MOVE_DAMP)) < 1e-6  # 3.3 px/tick


def test_fall_speed_is_capped():
    s = run((200.0, 0.0, 0, 0), (0, 0), level(), 40)
    assert s[3] == MAX_FALL


def test_no_tunnelling_through_thin_floor_at_max_fall():
    lv = level(solid=[(30, 60, 60, 60)])  # 5px ledge at y=300
    s = run((200.0, 3.0, 0.0, MAX_FALL), (0, 0), lv, 100)
    assert s[1] == 300 - PLAYER


def test_no_tunnelling_through_thin_wall_at_top_speed():
    lv = level(solid=[FLOOR, (60, 0, 60, 89)])
    s = run((100.0, GROUND, 0, 0), (1, 0), lv, 300)
    assert s[0] + PLAYER <= 300


def test_jump_is_fixed_height():
    lv = level(solid=[FLOOR])
    apexes = []
    for hold in (1, 5, 200):  # releasing early must not change the height
        s = step((200.0, GROUND, 0, 0), (0, 1), lv)
        assert s[3] < 0
        top = s[1]
        for t in range(60):
            s = step(s, (0, 1 if t < hold and s[1] < GROUND else 0), lv)
            top = min(top, s[1])
        apexes.append(GROUND - top)
    assert apexes[0] == apexes[1] == apexes[2] and 50 < apexes[0] < 62  # ~55 px
    assert JUMP_VEL == -10.0


def test_releasing_in_air_keeps_momentum_but_drags_more_than_holding():
    lv = level(solid=[FLOOR])
    s = run((50.0, GROUND, 0, 0), (1, 0), lv, 30)       # run to top speed
    s = step(s, (1, 1), lv)                             # jump
    held = run(s, (1, 0), lv, 10)
    released = run(s, (0, 0), lv, 10)
    assert held[2] > released[2] > 0.5 * s[2]           # keeps most speed, but less than holding
    assert abs(released[2] - s[2] * AIR_DRAG ** 10) < 1e-9


def test_ground_friction_stops_quickly():
    lv = level(solid=[FLOOR])
    s = run((50.0, GROUND, 0, 0), (1, 0), lv, 30)
    s = run(s, (0, 0), lv, 15)
    assert s[2] == 0.0


def test_spike_kills_and_respawns():
    lv = level(solid=[FLOOR], spike=[(30, 89, 60, 89)])
    s = (200.0, 400.0, 0, 0)
    for _ in range(100):
        ns, dead, s, _ = advance(s, (0, 0), lv)
        if dead:
            assert s[:2] == lv.spawn and ns[:2] != lv.spawn
            return
    raise AssertionError("never died")
