"""Play the real engine with the keyboard. Every tick is logged to data/human/.

    Left/Right or A/D   move          Up / Space / W   jump (hold to bunny-hop, like the Java game)
    R   restart current room (saves the episode, starts a new one; after a finish, restarts the run)
    N / P   skip to next / previous room (dev; also starts a new episode)
    Esc   quit (saves)

Physics runs exactly one fixed tick per rendered frame at 100 Hz, never variable dt.
Usage: python game.py [--zone N] [--no-log]
"""
import argparse
from pathlib import Path

import pygame

from platformnet.level import load_levels, levels_hash
from platformnet.logger import DATA_DIR, EpisodeLogger
from platformnet.physics import (BOUNCE, EXIT, GRID_H, GRID_W, ICE, ONE_WAY,
                                 PLAYER, SOLID, SPIKE, TICK_HZ, TILE)
from platformnet.run import EV_DEATH, EV_FINISH, advance, spawn_state

ASSETS = Path(__file__).resolve().parent / "assets"
SCREEN = (GRID_W * TILE, GRID_H * TILE)  # 500 x 500
COLORS = {
    "bg": (0xBC, 0xDF, 0xFF), SOLID: (0, 0, 0), ICE: (120, 200, 240), BOUNCE: (240, 120, 60),
    ONE_WAY: (120, 90, 40), EXIT: (60, 200, 90), "hud": (0, 0, 0),
}


def render_room(level, spike_img):
    surf = pygame.Surface(SCREEN)
    surf.fill(COLORS["bg"])
    for r in range(GRID_H):
        for c in range(GRID_W):
            t = level.tile(c, r)
            if t == SPIKE:
                surf.blit(spike_img, (c * TILE, r * TILE))
            elif t in COLORS:
                h = 2 if t == ONE_WAY else TILE  # draw one-ways as thin ledges
                pygame.draw.rect(surf, COLORS[t], (c * TILE, r * TILE, TILE, h))
    return surf


def read_action(keys):
    d = int(keys[pygame.K_RIGHT] or keys[pygame.K_d]) - int(keys[pygame.K_LEFT] or keys[pygame.K_a])
    j = int(keys[pygame.K_UP] or keys[pygame.K_SPACE] or keys[pygame.K_w])
    return (d, j)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zone", type=int, default=1, help="starting room, 1-based")
    ap.add_argument("--no-log", action="store_true")
    args = ap.parse_args()

    levels = load_levels()
    pygame.init()
    screen = pygame.display.set_mode(SCREEN)
    pygame.display.set_caption("platformNet - real engine")
    font = pygame.font.SysFont("consolas", 14, bold=True)
    clock = pygame.time.Clock()
    # The sprite is drawn exactly the size of the hitbox (the Java game drew 40px over 5px).
    player_img = pygame.transform.smoothscale(pygame.image.load(ASSETS / "player.png").convert_alpha(), (PLAYER, PLAYER))
    spike_img = pygame.transform.smoothscale(pygame.image.load(ASSETS / "spike.png").convert_alpha(), (TILE, TILE))
    rooms = [render_room(lv, spike_img) for lv in levels]

    log = None if args.no_log else EpisodeLogger(DATA_DIR / "human", "human", levels_hash(levels))
    last_saved = ""

    def start(z):
        s = spawn_state(levels[z])
        if log:
            log.begin(z, s)
        return z, s

    def save(finished=False):
        nonlocal last_saved
        if log and (p := log.save(finished)):
            last_saved = p.name

    zone, state = start(max(0, min(len(levels) - 1, args.zone - 1)))
    ticks = deaths = 0
    finished, running = False, True
    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    running = False
                elif ev.key == pygame.K_r:
                    save()
                    if finished:
                        zone, ticks, deaths, finished = 0, 0, 0, False
                    zone, state = start(zone)
                elif ev.key in (pygame.K_n, pygame.K_p) and not finished:
                    save()
                    zone, state = start((zone + (1 if ev.key == pygame.K_n else -1)) % len(levels))

        if not finished:
            action = read_action(pygame.key.get_pressed())
            ns, nz, nstate, event = advance(zone, state, action, levels)
            if log:
                log.record(zone, state, action, ns, event)
            zone, state = nz, nstate
            ticks += 1
            deaths += event == EV_DEATH
            if event == EV_FINISH:
                finished = True
                save(finished=True)

        screen.blit(rooms[zone], (0, 0))
        screen.blit(player_img, (round(state[0]), round(state[1])))
        hud = f"{zone + 1}/{len(levels)} {levels[zone].name}   t={ticks / TICK_HZ:6.2f}s   deaths {deaths}"
        if log:
            hud += f"   REC {len(log)}"
        screen.blit(font.render(hud, True, COLORS["hud"]), (6, 4))
        if last_saved:
            screen.blit(font.render(f"saved {last_saved}", True, COLORS["hud"]), (6, 20))
        if finished:
            msg = font.render(f"FINISHED {ticks / TICK_HZ:.2f}s, {deaths} deaths - R to run again", True, COLORS["hud"])
            screen.blit(msg, msg.get_rect(center=(SCREEN[0] // 2, SCREEN[1] // 2)))
        pygame.display.flip()
        clock.tick(TICK_HZ)

    if not finished:
        save()
    pygame.quit()


if __name__ == "__main__":
    main()
