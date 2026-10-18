"""Behaviour checks on tiny hand-made levels."""
from platformnet.level import parse_level
from platformnet.physics import (AIR_RESISTANCE, FRICTION, GROUND_ACCEL, JUMP_VEL, MAX_FALL, MAX_RUN,
                                 PLAYER, SOLID, ground_friction, is_grounded, step)
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


FLOOR = (0, 90, 99, 99)
GROUND = 450.0 - PLAYER


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
    lv = level(solid=[FLOOR, (60, 0, 60, 89)])
    for x0 in (100.0, 250.0, 283.7):
        s = run((x0, GROUND, 0, 0), (1, 0), lv, 300)
        assert s[0] == 300 - PLAYER and s[2] == 0


def test_top_speed():
    s = run((0.0, GROUND, 0, 0), (1, 0), level(solid=[FLOOR]), 60)
    assert s[2] == MAX_RUN


def test_fall_speed_is_capped():
    s = run((200.0, 0.0, 0, 0), (0, 0), level(), 40)
    assert s[3] == MAX_FALL


def test_no_tunnelling_through_thin_floor_at_max_fall():
    lv = level(solid=[(30, 60, 60, 60)])
    s = run((200.0, 3.0, 0.0, MAX_FALL), (0, 0), lv, 100)
    assert s[1] == 300 - PLAYER


def test_no_tunnelling_through_thin_wall_at_top_speed():
    lv = level(solid=[FLOOR, (60, 0, 60, 89)])
    s = run((100.0, GROUND, 0, 0), (1, 0), lv, 300)
    assert s[0] + PLAYER <= 300


def test_jump_is_fixed_height():
    lv = level(solid=[FLOOR])
    apexes = []
    for hold in (1, 5, 200):
        s = step((200.0, GROUND, 0, 0), (0, 1), lv)
        assert s[3] < 0
        top = s[1]
        for t in range(60):
            s = step(s, (0, 1 if t < hold and s[1] < GROUND else 0), lv)
            top = min(top, s[1])
        apexes.append(GROUND - top)
    assert apexes[0] == apexes[1] == apexes[2] and 50 < apexes[0] < 60


def test_releasing_in_air_keeps_momentum_but_drags_more_than_holding():
    lv = level(solid=[FLOOR])
    s = run((50.0, GROUND, 0, 0), (1, 0), lv, 30)
    s = step(s, (1, 1), lv)
    held = run(s, (1, 0), lv, 10)
    released = run(s, (0, 0), lv, 10)
    assert held[2] > released[2] > 0.5 * s[2]
    assert abs(released[2] - s[2] * (1 - AIR_RESISTANCE) ** 10) < 1e-9
    assert abs(held[2] - MAX_RUN) < 1e-9


def test_ground_decelerates_at_mu_times_ground_accel():
    lv = level(solid=[FLOOR])
    s = run((50.0, GROUND, 0, 0), (1, 0), lv, 30)
    s = step(s, (0, 0), lv)
    assert abs(s[2] - (MAX_RUN - FRICTION[SOLID] * GROUND_ACCEL)) < 1e-12
    s = run(s, (0, 0), lv, 10)
    assert s[2] == 0.0


def test_material_friction_controls_grip(monkeypatch):
    lv = level(solid=[FLOOR])
    def ticks_to_top_speed():
        s, n = (50.0, GROUND, 0, 0), 0
        while s[2] < MAX_RUN:
            s, n = step(s, (1, 0), lv), n + 1
        return n
    normal = ticks_to_top_speed()
    monkeypatch.setitem(FRICTION, SOLID, 0.25)
    assert ground_friction(50.0, GROUND, lv) == 0.25
    assert ticks_to_top_speed() > 3 * normal


def test_spike_kills_and_respawns():
    lv = level(solid=[FLOOR], spike=[(30, 89, 60, 89)])
    s = (200.0, 400.0, 0, 0)
    for _ in range(100):
        ns, dead, s, _ = advance(s, (0, 0), lv)
        if dead:
            assert s[:2] == lv.spawn and ns[:2] != lv.spawn
            return
    raise AssertionError("never died")
