"""Look at what the generator makes, and measure it.

    python preview_maps.py [seed]

Saves maps_preview.png (16 maps, 4x4) next to this file, then prints how often
each tile type appears over 1000 maps and how long generation takes.
"""
import sys
import time
from pathlib import Path

import pygame
import torch

from map_gen import make_maps
from true_view import N_TYPES, to_rgb

NAMES = ["empty", "solid", "exit", "spike"]
OUT = Path(__file__).resolve().parent / "maps_preview.png"


def save_grid(maps, path, n=4, scale=2, gap=6):
    size = 100 * scale
    sheet = pygame.Surface((n * size + (n + 1) * gap, n * size + (n + 1) * gap))
    sheet.fill((90, 90, 100))
    for k in range(n * n):
        rgb = to_rgb(maps[k]).numpy().transpose(1, 0, 2)          # pygame wants [x, y, 3]
        img = pygame.transform.scale(pygame.surfarray.make_surface(rgb), (size, size))
        sheet.blit(img, (gap + (k % n) * (size + gap), gap + (k // n) * (size + gap)))
    pygame.image.save(sheet, str(path))


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    save_grid(make_maps(16, seed=seed), OUT)
    print(f"saved {OUT.name}")

    t = time.perf_counter()
    maps = make_maps(1000, seed=seed + 1)
    ms = (time.perf_counter() - t) * 1000
    counts = torch.bincount(maps.flatten(), minlength=N_TYPES).double()
    print("tile mix over 1000 maps: " + ", ".join(
        f"{name} {100 * c / counts.sum():.2f}%" for name, c in zip(NAMES, counts)))
    has = [(maps == code).flatten(1).any(1).float().mean().item() for code in range(N_TYPES)]
    print("maps containing each type: " + ", ".join(f"{n} {100 * h:.0f}%" for n, h in zip(NAMES, has)))
    print(f"speed: {ms / 1000 * 256:.1f} ms per batch of 256")


if __name__ == "__main__":
    main()
