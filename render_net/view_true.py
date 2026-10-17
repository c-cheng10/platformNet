"""Look at true_view() by eye: a camera driven around the map.

    Tab        switch mode: FREE camera <-> PLAY (real physics, camera follows the player)
    Arrows     FREE: move the camera (Shift = fast)   PLAY: run / jump (Up or Space)
    R          PLAY: respawn
    Esc        quit

Left: the 150x150 camera view from true_view(), scaled 3x.
Right: the whole map with the camera's area outlined.
Usage: python view_true.py [level]   (e.g. level2; default <repo>/levels/level1.txt)
"""
import sys

import pygame
import torch

from true_view import PALETTE, VIEW, load_tiles, to_rgb, true_view  # also puts game_py on the path
from platformnet.level import load_level  # noqa: E402
from platformnet.physics import GRID_W, PLAYER, TILE  # noqa: E402
from platformnet.run import advance, spawn_state  # noqa: E402

SCALE = 3
MAP_PX = GRID_W * TILE            # 500
MINI = 250                        # minimap size on screen
W, H = VIEW * SCALE + 10 + MINI, VIEW * SCALE + 24


def surface_from_types(types):
    rgb = to_rgb(types).numpy()                        # [H, W, 3]
    return pygame.surfarray.make_surface(rgb.transpose(1, 0, 2))


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else None
    tiles = load_tiles(path)
    level = load_level(path)

    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("render_net - true_view")
    font = pygame.font.SysFont("consolas", 13, bold=True)
    clock = pygame.time.Clock()

    whole_map = pygame.transform.scale(
        pygame.surfarray.make_surface(PALETTE[tiles].numpy().transpose(1, 0, 2)), (MINI, MINI))

    mode = "FREE"
    cam = [250.0, 250.0]
    state = spawn_state(level)
    running = True
    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE):
                running = False
            elif ev.type == pygame.KEYDOWN and ev.key == pygame.K_TAB:
                mode = "PLAY" if mode == "FREE" else "FREE"
            elif ev.type == pygame.KEYDOWN and ev.key == pygame.K_r:
                state = spawn_state(level)

        keys = pygame.key.get_pressed()
        if mode == "FREE":
            speed = 3.0 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0.3
            cam[0] += speed * (keys[pygame.K_RIGHT] - keys[pygame.K_LEFT])
            cam[1] += speed * (keys[pygame.K_DOWN] - keys[pygame.K_UP])
        else:
            action = (keys[pygame.K_RIGHT] - keys[pygame.K_LEFT], int(keys[pygame.K_UP] or keys[pygame.K_SPACE]))
            _, _, state, _ = advance(state, action, level)
            cam = [state[0] + PLAYER / 2, state[1] + PLAYER / 2]

        view = true_view(tiles, torch.tensor([cam[0]]), torch.tensor([cam[1]]))[0]
        screen.fill((30, 30, 36))
        screen.blit(pygame.transform.scale(surface_from_types(view), (VIEW * SCALE, VIEW * SCALE)), (0, 0))
        if mode == "PLAY":  # player is always at the centre of its own camera
            c = (VIEW // 2 - PLAYER // 2) * SCALE
            pygame.draw.rect(screen, (255, 255, 255), (c, c, PLAYER * SCALE, PLAYER * SCALE))
            pygame.draw.rect(screen, (0, 0, 0), (c, c, PLAYER * SCALE, PLAYER * SCALE), 1)
        else:
            m = VIEW * SCALE // 2
            pygame.draw.line(screen, (255, 160, 0), (m - 6, m), (m + 6, m))
            pygame.draw.line(screen, (255, 160, 0), (m, m - 6), (m, m + 6))

        ox = VIEW * SCALE + 10
        screen.blit(whole_map, (ox, 0))
        k = MINI / MAP_PX
        pygame.draw.rect(screen, (255, 160, 0),
                         (ox + (cam[0] - VIEW / 2) * k, (cam[1] - VIEW / 2) * k, VIEW * k, VIEW * k), 1)
        info = f"{mode}  camera ({cam[0]:.2f}, {cam[1]:.2f})   Tab: switch mode"
        screen.blit(font.render(info, True, (220, 220, 220)), (6, VIEW * SCALE + 5))
        pygame.display.flip()
        clock.tick(60)
    pygame.quit()


if __name__ == "__main__":
    main()
