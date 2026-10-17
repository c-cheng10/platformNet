"""Play the game with the keyboard.

    Left/Right or A/D   move          Up / Space / W   jump (fixed height; hold to hop again on landing)
    R   restart
    Esc   quit

Physics runs exactly one fixed tick per rendered frame at 60 Hz, never variable dt.
Usage: python game.py [level file]   (default levels/random2.txt; make levels with level_editor.py)
Every run is recorded to recordings/ (see gamerecorder.py).
"""
import sys
from pathlib import Path

import pygame

from platformnet.level import load_level
from platformnet.physics import EXIT, GRID_H, GRID_W, PLAYER, SOLID, SPIKE, TICK_HZ, TILE
from platformnet.run import advance, spawn_state

ASSETS = Path(__file__).resolve().parent / "assets"
SCREEN = (GRID_W * TILE, GRID_H * TILE)  # 500 x 500
COLORS = {"bg": (0xBC, 0xDF, 0xFF), SOLID: (0, 0, 0), EXIT: (60, 200, 90), "hud": (0, 0, 0)}


def render_level(level, spike_img):
    surf = pygame.Surface(SCREEN)
    surf.fill(COLORS["bg"])
    for r in range(GRID_H):
        for c in range(GRID_W):
            t = level.tile(c, r)
            if t == SPIKE:
                surf.blit(spike_img, (c * TILE, r * TILE))
            elif t in COLORS:
                pygame.draw.rect(surf, COLORS[t], (c * TILE, r * TILE, TILE, TILE))
    return surf


def read_action(keys):
    d = int(keys[pygame.K_RIGHT] or keys[pygame.K_d]) - int(keys[pygame.K_LEFT] or keys[pygame.K_a])
    j = int(keys[pygame.K_UP] or keys[pygame.K_SPACE] or keys[pygame.K_w])
    return (d, j)


def main(level_path=None, recorder=None):
    """recorder (see gamerecorder.py) gets every tick's transition and one end_episode() per run."""
    level = load_level(level_path) if level_path else load_level()
    pygame.init()
    screen = pygame.display.set_mode(SCREEN)
    pygame.display.set_caption("platformNet - real engine")
    font = pygame.font.SysFont("consolas", 14, bold=True)
    clock = pygame.time.Clock()
    # The sprite is drawn exactly the size of the hitbox (the Java game drew 40px over 5px).
    player_img = pygame.transform.smoothscale(pygame.image.load(ASSETS / "player.png").convert_alpha(), (PLAYER, PLAYER))
    spike_img = pygame.transform.smoothscale(pygame.image.load(ASSETS / "spike.png").convert_alpha(), (TILE, TILE))
    background = render_level(level, spike_img)

    state, finished = spawn_state(level), False
    running = True
    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE):
                running = False
            elif ev.type == pygame.KEYDOWN and ev.key == pygame.K_r:
                if recorder:
                    recorder.end_episode()
                state, finished = spawn_state(level), False

        if not finished:
            action = read_action(pygame.key.get_pressed())
            _, dead, new_state, finished = advance(state, action, level)
            if recorder:
                recorder.record(state, action, new_state, dead, finished)
            state = new_state
            if finished and recorder:
                recorder.end_episode()

        screen.blit(background, (0, 0))
        screen.blit(player_img, (round(state[0]), round(state[1])))
        if finished:
            msg = font.render("FINISHED - R to run again", True, COLORS["hud"])
            screen.blit(msg, msg.get_rect(center=(SCREEN[0] // 2, SCREEN[1] // 2)))
        pygame.display.flip()
        clock.tick(TICK_HZ)

    if recorder:
        recorder.end_episode()
    pygame.quit()


if __name__ == "__main__":
    from gamerecorder import GameRecorder

    level_path = sys.argv[1] if len(sys.argv) > 1 else None
    main(level_path, GameRecorder(level_path) if level_path else GameRecorder())
