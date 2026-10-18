"""Draw levels with the mouse and save them as the ASCII files the game loads.

    Left mouse      paint with the selected tile      Right mouse or Ctrl+left   erase
    Shift + drag    fill a rectangle (left = tile, right or Ctrl = erase)
    1 solid   2 spike   3 exit   4 spawn (one per level; placing it moves it)
    [ / ]           brush size smaller / bigger
    G               toggle grid lines
    Ctrl+Z          undo        Ctrl+S   save        Ctrl+N   clear
    Esc             quit (unsaved changes are lost)

Usage: python level_editor.py [level]   e.g. level4 -> <repo>/levels/level4.txt (default level2;
created if missing). A full path also works.
The grid is 100 x 100 tiles, the same as the game. The white box at the spawn
shows the player's 2x2-tile size.
"""
import sys
from pathlib import Path

import pygame

# the engine (platformnet) lives in <repo>/game_py
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "game_py"))
from platformnet.level import parse_level, resolve_level  # noqa: E402
from platformnet.physics import GRID_H, GRID_W, TILE

CELL = 7                       # on-screen pixels per tile (the game uses 5)
BAR = 34                       # status bar height
W, H = GRID_W * CELL, GRID_H * CELL
TOOLS = [("#", "solid", (0, 0, 0)), ("X", "spike", (220, 40, 40)),
         ("E", "exit", (60, 200, 90)), ("S", "spawn", (255, 255, 255))]
COLOR = {ch: col for ch, _, col in TOOLS}
BG = (0xBC, 0xDF, 0xFF)
MAX_UNDO = 50


def load_grid(path):
    """Load a level; a wrong-sized file is padded/trimmed to 100x100 (never silently blanked,
    so Ctrl+S can't wipe a real level). Returns (grid, warning or "")."""
    if not path.exists():
        return blank_grid(), f"new level {path.name}"
    rows = [l.rstrip("\r\n") for l in path.read_text().splitlines() if l.strip() and not l.startswith(";")]
    warning = ""
    if len(rows) != GRID_H or any(len(r) != GRID_W for r in rows):
        warning = f"{path.name} was {max(map(len, rows), default=0)}x{len(rows)}: padded/trimmed to {GRID_W}x{GRID_H}, check it"
        print(warning)
    grid = [list(r[:GRID_W].ljust(GRID_W, ".")) for r in rows[:GRID_H]]
    grid += [["."] * GRID_W for _ in range(GRID_H - len(grid))]
    return grid, warning


def blank_grid():
    g = [["."] * GRID_W for _ in range(GRID_H)]
    for c in range(GRID_W):          # a floor and a spawn so a new level is playable
        for r in range(GRID_H - 5, GRID_H):
            g[r][c] = "#"
    g[GRID_H - 7][3] = "S"
    return g


def save_grid(grid, path):
    """Validate with the game's own loader, then write. Returns an error string or None."""
    text = "\n".join("".join(r) for r in grid) + "\n"
    try:
        level = parse_level(text)
    except ValueError as e:
        return str(e)
    c, r = (int(v) // TILE for v in level.spawn)
    if any(grid[rr][cc] not in ".S" for rr in (r, r + 1) for cc in (c, c + 1)):
        return "spawn overlaps a block (the player is 2x2 tiles)"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return None


def cell_at(pos):
    x, y = pos
    if 0 <= x < W and 0 <= y < H:
        return x // CELL, y // CELL
    return None


def paint(grid, c, r, ch, size):
    if ch == "S":
        set_spawn(grid, c, r)
        return
    for rr in range(r - size // 2, r - size // 2 + size):
        for cc in range(c - size // 2, c - size // 2 + size):
            if 0 <= cc < GRID_W and 0 <= rr < GRID_H:
                grid[rr][cc] = ch


def fill_rect(grid, a, b, ch):
    if ch == "S":
        set_spawn(grid, *b)
        return
    for r in range(min(a[1], b[1]), max(a[1], b[1]) + 1):
        for c in range(min(a[0], b[0]), max(a[0], b[0]) + 1):
            grid[r][c] = ch


def set_spawn(grid, c, r):
    for row in grid:
        for i, ch in enumerate(row):
            if ch == "S":
                row[i] = "."
    grid[min(r, GRID_H - 2)][min(c, GRID_W - 2)] = "S"


def draw(screen, font, grid, tool, size, show_grid, rect_drag, mouse_cell, msg, dirty):
    screen.fill(BG)
    spawn = None
    for r, row in enumerate(grid):
        for c, ch in enumerate(row):
            if ch == "S":
                spawn = (c, r)
            elif ch in COLOR:
                pygame.draw.rect(screen, COLOR[ch], (c * CELL, r * CELL, CELL, CELL))
    if spawn:  # the player is 2x2 tiles
        pygame.draw.rect(screen, (255, 255, 255), (spawn[0] * CELL, spawn[1] * CELL, 2 * CELL, 2 * CELL))
        pygame.draw.rect(screen, (0, 0, 0), (spawn[0] * CELL, spawn[1] * CELL, 2 * CELL, 2 * CELL), 1)
    if show_grid:
        for i in range(0, GRID_W + 1, 10):
            pygame.draw.line(screen, (150, 180, 210), (i * CELL, 0), (i * CELL, H))
            pygame.draw.line(screen, (150, 180, 210), (0, i * CELL), (W, i * CELL))
    if rect_drag and mouse_cell:
        (c0, r0), (c1, r1) = rect_drag[0], mouse_cell
        rect = (min(c0, c1) * CELL, min(r0, r1) * CELL, (abs(c1 - c0) + 1) * CELL, (abs(r1 - r0) + 1) * CELL)
        pygame.draw.rect(screen, (255, 160, 0), rect, 2)
    elif mouse_cell and tool != "S":
        c, r = mouse_cell
        o = size // 2
        pygame.draw.rect(screen, (255, 160, 0), ((c - o) * CELL, (r - o) * CELL, size * CELL, size * CELL), 1)

    pygame.draw.rect(screen, (40, 40, 48), (0, H, W, BAR))
    x = 6
    for i, (ch, name, col) in enumerate(TOOLS, 1):
        label = font.render(f"{i} {name}", True, (255, 255, 255))
        if ch == tool:
            pygame.draw.rect(screen, (90, 90, 110), (x - 3, H + 5, label.get_width() + 6, BAR - 10))
        pygame.draw.rect(screen, col, (x, H + 12, 8, 8))
        screen.blit(label, (x + 12, H + 9))
        x += label.get_width() + 26
    pos = f"  ({mouse_cell[0]},{mouse_cell[1]})" if mouse_cell else ""
    info = f"brush {size}{pos}   {msg or ('unsaved' if dirty else 'saved')}"
    screen.blit(font.render(info, True, (255, 220, 120) if msg else (200, 200, 200)), (x + 4, H + 9))


def main():
    path = resolve_level(sys.argv[1] if len(sys.argv) > 1 else None)
    grid, load_warning = load_grid(path)
    pygame.init()
    screen = pygame.display.set_mode((W, H + BAR))
    pygame.display.set_caption(f"platformNet level editor - {path.name}")
    font = pygame.font.SysFont("consolas", 13, bold=True)
    clock = pygame.time.Clock()

    tool, size, show_grid = "#", 1, True
    undo, dirty, msg, msg_until = [], False, "", 0
    rect_drag = None      # (start_cell, char) while shift-dragging
    painting = None       # char being painted while dragging

    def push_undo():
        undo.append([row[:] for row in grid])
        del undo[:-MAX_UNDO]

    def flash(text):
        nonlocal msg, msg_until
        msg, msg_until = text, pygame.time.get_ticks() + 3000

    if load_warning:
        flash(load_warning)
        msg_until += 7000  # keep a load problem on screen for 10 s

    running = True
    while running:
        mouse_cell = cell_at(pygame.mouse.get_pos())
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                ctrl = ev.mod & pygame.KMOD_CTRL
                if ev.key == pygame.K_ESCAPE:
                    running = False
                elif ctrl and ev.key == pygame.K_s:
                    err = save_grid(grid, path)
                    if err:
                        flash(f"NOT saved: {err}")
                    else:
                        dirty = False
                        flash(f"saved {path.name}")
                elif ctrl and ev.key == pygame.K_z:
                    if undo:
                        grid[:] = undo.pop()
                        dirty = True
                elif ctrl and ev.key == pygame.K_n:
                    push_undo()
                    grid[:] = blank_grid()
                    dirty = True
                elif pygame.K_1 <= ev.key <= pygame.K_4:
                    tool = TOOLS[ev.key - pygame.K_1][0]
                elif ev.key == pygame.K_LEFTBRACKET:
                    size = max(1, size - 1)
                elif ev.key == pygame.K_RIGHTBRACKET:
                    size = min(20, size + 1)
                elif ev.key == pygame.K_g:
                    show_grid = not show_grid
            elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button in (1, 3) and mouse_cell:
                ctrl_held = pygame.key.get_mods() & pygame.KMOD_CTRL
                ch = tool if ev.button == 1 and not ctrl_held else "."   # Ctrl+click = erase
                push_undo()
                dirty = True
                if pygame.key.get_mods() & pygame.KMOD_SHIFT:
                    rect_drag = (mouse_cell, ch)
                else:
                    painting = ch
                    paint(grid, *mouse_cell, ch, size)
            elif ev.type == pygame.MOUSEBUTTONUP and ev.button in (1, 3):
                if rect_drag and mouse_cell:
                    fill_rect(grid, rect_drag[0], mouse_cell, rect_drag[1])
                rect_drag, painting = None, None

        if painting and mouse_cell and painting != "S":
            paint(grid, *mouse_cell, painting, size)
        if msg and pygame.time.get_ticks() > msg_until:
            msg = ""

        draw(screen, font, grid, tool, size, show_grid, rect_drag, mouse_cell, msg, dirty)
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()
