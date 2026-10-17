"""Play the real engine with the keyboard. Every tick is logged to data/human/.

    Left/Right or A/D   move          Up / Space / W   jump (hold to hop on landing, like the Java game)
    R   restart (saves the episode, starts a new one)
    Esc   quit (saves)

Physics runs exactly one fixed tick per rendered frame at 100 Hz, never variable dt.
Usage: python game.py [--no-log]
"""
import argparse
from pathlib import Path

import pygame

from platformnet.level import load_level
from platformnet.logger import DATA_DIR, EpisodeLogger
from platformnet.physics import EXIT, GRID_H, GRID_W, PLAYER, SOLID, SPIKE, TICK_HZ, TILE
from platformnet.run import EV_FINISH, advance, spawn_state

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-log", action="store_true")
    args = ap.parse_args()

    level = load_level()
    pygame.init()
    screen = pygame.display.set_mode(SCREEN)
    pygame.display.set_caption("platformNet - real engine")
    font = pygame.font.SysFont("consolas", 14, bold=True)
    clock = pygame.time.Clock()
    # The sprite is drawn exactly the size of the hitbox (the Java game drew 40px over 5px).
    player_img = pygame.transform.smoothscale(pygame.image.load(ASSETS / "player.png").convert_alpha(), (PLAYER, PLAYER))
    spike_img = pygame.transform.smoothscale(pygame.image.load(ASSETS / "spike.png").convert_alpha(), (TILE, TILE))
    background = render_level(level, spike_img)

    log = None if args.no_log else EpisodeLogger(DATA_DIR / "human", "human", level.hash)
    last_saved = ""

    def save(finished=False):
        nonlocal last_saved
        if log and (p := log.save(finished)):
            last_saved = p.name

    def restart():
        s = spawn_state(level)
        if log:
            log.begin(s)
        return s, False

    state, finished = restart()
    running = True
    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE):
                running = False
            elif ev.type == pygame.KEYDOWN and ev.key == pygame.K_r:
                save()
                state, finished = restart()

        if not finished:
            action = read_action(pygame.key.get_pressed())
            ns, nstate, event = advance(state, action, level)
            if log:
                log.record(state, action, ns, event)
            state = nstate
            if event == EV_FINISH:
                finished = True
                save(finished=True)

        screen.blit(background, (0, 0))
        screen.blit(player_img, (round(state[0]), round(state[1])))
        if log:
            screen.blit(font.render(f"REC {len(log)}", True, COLORS["hud"]), (6, 4))
        if last_saved:
            screen.blit(font.render(f"saved {last_saved}", True, COLORS["hud"]), (6, 20))
        if finished:
            msg = font.render("FINISHED - R to run again", True, COLORS["hud"])
            screen.blit(msg, msg.get_rect(center=(SCREEN[0] // 2, SCREEN[1] // 2)))
        pygame.display.flip()
        clock.tick(TICK_HZ)

    if not finished:
        save()
    pygame.quit()


if __name__ == "__main__":
    main()
