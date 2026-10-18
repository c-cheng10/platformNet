"""Watch the world model play a level: it plans with the neural network, the real engine executes.

    ./demo.sh level1                      live window
    ./demo.sh level2 --gif demo.gif       same, and also save everything shown live as a GIF
    ./demo.sh level2 --gif demo.gif --headless   just write the GIF (no window)
    ./demo.sh level2 --execute 60         run 60 ticks of each plan before replanning (default 30)

The loop, like a model-predictive controller:
  1. Plan: from the player's real state, a best-first search over held key combos where every candidate
     future is imagined by the network (hundreds at once, batched). The engine is never consulted here.
  2. Execute: feed the first --execute ticks of that plan's keys to the real engine (game_py/platformnet).
  3. Replan from wherever the engine actually put the player. Also replan early if reality drifts more than
     REPLAN_DIST from what the model predicted, and after a death.

On screen: while the model plans, every future it imagines appears as a faint blue dot and its best path
so far as a green line; once it commits, the green line is the path it expects. Every path it drops
stays behind as a faint red line until that plan has been executed. The yellow trail is where the player
really went. When the level is done, the window loops a clean replay of just the real run until you close it.
"""
import argparse
import heapq
from collections import deque
import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))
sys.path.insert(0, str(ROOT / "game_py"))
from model import CROP, HALF, NextStatePredictor
from train import PAD, normalize_extra, pad_level, parse_level
from platformnet.level import load_level, resolve_level
from platformnet.run import advance, spawn_state

MODEL_PATH = ROOT / "models" / "model_b128_e100_noise2_p30.pt"

TICK_HZ = 60
TILE = 5
PLAYER = 10
LEVEL_PX = 100 * TILE
SOLID, EXIT, SPIKE = 1, 2, 3
MAX_RUN = 3.3
UNREACHABLE = 1e4

ACTIONS = [(d, j) for j in (0, 1) for d in (1, 0, -1)]
HOLD = 4
BATCH = 64
MAX_STEPS = 500
EXECUTE = 30
REPLAN_DIST = 2.0
MAX_TICKS = 60 * TICK_HZ
MAX_DEATHS = 5
SCALE = 2


class WorldModel:
    """The trained network, stepping a whole batch of (state, keys) at once."""

    def __init__(self, text, model_path):
        ckpt = torch.load(model_path, map_location="cpu")
        self.model = NextStatePredictor(ckpt["hidden_size"])
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()
        self.stats = ckpt["stats"]
        self.grid = parse_level(text).long()
        self.padded = pad_level(self.grid.to(torch.uint8))
        self.dist = self.route_distance()

    @torch.no_grad()
    def step(self, state, keys):
        """state (B,4) [x, y, vx, vy], keys (B,3) [left, right, jump] -> next state (B,4), as in train.load_predictor."""
        tile = torch.floor(state[:, :2] / TILE).long()
        ar = torch.arange(CROP)
        rows = (tile[:, 1:] + PAD - HALF) + ar
        cols = (tile[:, :1] + PAD - HALF) + ar
        crops = self.padded[rows[:, :, None], cols[:, None, :]]
        extra = torch.cat([keys, (state[:, :2] % TILE) / TILE, state[:, 2:]], dim=1)
        out = state + self.model(crops, normalize_extra(extra, self.stats)) * self.stats["delta_std"] + self.stats["delta_mean"]
        pos = out[:, :2].clamp(0, LEVEL_PX - PLAYER)
        vel = torch.where(pos == out[:, :2], out[:, 2:], 0.0)
        return torch.cat([pos, vel], dim=1)

    def touching(self, state, code):
        """(B,) True where the player's box strictly overlaps a tile of `code` (render.py's overlaps())."""
        hit = torch.zeros(len(state), dtype=torch.bool)
        r0 = torch.floor(state[:, 1] / TILE).long()
        c0 = torch.floor(state[:, 0] / TILE).long()
        r1 = torch.ceil((state[:, 1] + PLAYER) / TILE).long()
        c1 = torch.ceil((state[:, 0] + PLAYER) / TILE).long()
        for dr in range(3):
            for dc in range(3):
                r, c = r0 + dr, c0 + dc
                ok = (r < r1) & (c < c1) & (r >= 0) & (r < 100) & (c >= 0) & (c < 100)
                hit |= ok & (self.grid[r.clamp(0, 99), c.clamp(0, 99)] == code)
        return hit

    def measure_jump(self):
        """Ask the model how high the player can jump: imagine one jump on an empty flat floor.
        Returns (apex height in px, gravity in px/tick^2) as the network sees them."""
        flat = torch.zeros(100, 100, dtype=torch.uint8)
        flat[90:] = SOLID
        padded, self.padded = self.padded, pad_level(flat)
        state = torch.tensor([[250.0, 90 * TILE - PLAYER, 0.0, 0.0]])
        ys, vys = [], []
        for t in range(40):
            state = self.step(state, torch.tensor([[0.0, 0.0, float(t == 0)]]))  # tap jump once
            ys.append(state[0, 1].item())
            vys.append(state[0, 3].item())
        self.padded = padded
        rising = [b - a for a, b in zip(vys, vys[1:]) if a < 0 and b < 0]
        return 90 * TILE - PLAYER - min(ys), sum(rising) / len(rising)

    def route_distance(self):
        """(n, n, J+1) moves from each player position (its top-left tile) to the exit, for each amount of
        jump height left (in tiles), through a tile graph that respects what jumping can do:
          - standing on solid refills the budget to a full jump (J tiles, measured from the model)
          - while there is budget left you can go up, up-diagonally or sideways, each costing one tile
          - with none left you can only fall: down or down-diagonally
        Plain open-space distance would route straight up walls and shafts, which no jump can follow."""
        apex, self.gravity = self.measure_jump()
        J = self.jump_tiles = int(apex // TILE)
        k = PLAYER // TILE
        n = 100 - k + 1
        blocked = (self.grid == SOLID) | (self.grid == SPIKE)
        solid = torch.cat([self.grid == SOLID, torch.ones(1, 100, dtype=torch.bool)])  # below the grid is solid
        fits = torch.zeros(n, n, dtype=torch.bool)
        at_exit = torch.zeros(n, n, dtype=torch.bool)
        ground = torch.zeros(n, n, dtype=torch.bool)
        for dr in range(k):
            for dc in range(k):
                fits |= blocked[dr:dr + n, dc:dc + n]
                at_exit |= (self.grid == EXIT)[dr:dr + n, dc:dc + n]
        for dc in range(k):
            ground |= solid[k:k + n, dc:dc + n]
        fits = (~fits).tolist()
        ground = [[g and f for g, f in zip(gr, fr)] for gr, fr in zip(ground.tolist(), fits)]
        self.ground = ground

        def ok(r, c, dr, dc):
            nr, nc = r + dr, c + dc
            if not (0 <= nr < n and 0 <= nc < n and fits[nr][nc]):
                return False
            return dr == 0 or dc == 0 or fits[r + dr][c] or fits[r][c + dc]  # don't squeeze through corners

        # Build the graph backwards (who can reach me), then BFS out from the exit
        preds = {}
        for r in range(n):
            for c in range(n):
                if not fits[r][c]:
                    continue
                for j in ([J] if ground[r][c] else range(J + 1)):
                    moves = [(1, 0, 0), (1, -1, 0), (1, 1, 0)]  # falling
                    if ground[r][c]:
                        moves += [(0, -1, None), (0, 1, None), (-1, 0, J - 1), (-1, -1, J - 1), (-1, 1, J - 1)]
                    elif j > 0:
                        moves += [(0, -1, j - 1), (0, 1, j - 1), (-1, 0, j - 1), (-1, -1, j - 1), (-1, 1, j - 1)]
                    for dr, dc, nj in moves:
                        if not ok(r, c, dr, dc):
                            continue
                        nr, nc = r + dr, c + dc
                        nj = J if ground[nr][nc] else (0 if nj is None else nj)  # walking off a ledge = falling
                        preds.setdefault((nr, nc, nj), []).append((r, c, j))
        dist = torch.full((n, n, J + 1), UNREACHABLE)
        queue = deque()
        for r, c in (at_exit & torch.tensor(fits)).nonzero().tolist():
            for j in range(J + 1):
                dist[r, c, j] = 0
                queue.append((r, c, j))
        d = dist.tolist()
        while queue:
            node = queue.popleft()
            r, c, j = node
            for pr, pc, pj in preds.get(node, ()):
                if d[pr][pc][pj] == UNREACHABLE:
                    d[pr][pc][pj] = d[r][c][j] + 1
                    queue.append((pr, pc, pj))
        return torch.tensor(d)

    def heuristic(self, state):
        """Rough ticks to the exit along a route jumps can actually follow, at full run speed. The jump
        budget comes from the state: a full jump if standing, what's left of the rise if going up, else none.
        Looks at the tiles around the player's top-left, so a state a hair inside a wall still gets a score."""
        n, J = self.dist.shape[0], self.jump_tiles
        r = torch.floor(state[:, 1] / TILE).long()
        c = torch.floor(state[:, 0] / TILE).long()
        vy = state[:, 3]
        rise = torch.where(vy < 0, vy * vy / (2 * self.gravity), 0.0)
        j = torch.clamp(torch.floor(rise / TILE).long(), 0, J)
        best = torch.full((len(state),), UNREACHABLE)
        for dr in (0, 1):
            for dc in (0, 1):
                rr, cc = (r + dr).clamp(0, n - 1), (c + dc).clamp(0, n - 1)
                standing = torch.tensor([self.ground[a][b] for a, b in zip(rr.tolist(), cc.tolist())]) & (vy.abs() < 1)
                best = torch.minimum(best, self.dist[rr, cc, torch.where(standing, J, j)])
        return best * TILE / MAX_RUN


def plan(wm, start):
    """Batched best-first search from `start`, every step imagined by the network.
    A generator: after each search step it yields ("search", start, (imagined positions, best path so far))
    so the search can be drawn live, then returns ([(predicted_state, action), ...] per tick, reached_exit,
    search steps). If the exit isn't reached within MAX_STEPS, returns the path to the imagined state that
    got closest to it."""
    nodes = [(None, None, [torch.tensor(start, dtype=torch.float32)])]
    frontier = [(0.0, 0, 0)]
    seen = set()
    best, best_h = 0, float("inf")
    for it in range(MAX_STEPS):
        if not frontier:
            break
        batch = [heapq.heappop(frontier) for _ in range(min(BATCH, len(frontier)))]
        state = torch.stack([nodes[i][2][-1] for _, _, i in batch]).repeat_interleave(len(ACTIONS), 0)
        keys = torch.tensor([[d < 0, d > 0, j] for d, j in ACTIONS], dtype=torch.float32).repeat(len(batch), 1)
        alive = torch.ones(len(state), dtype=torch.bool)
        won = torch.zeros(len(state), dtype=torch.long)
        trace = []
        for t in range(HOLD):
            state = torch.where((alive & (won == 0))[:, None], wm.step(state, keys), state)
            trace.append(state)
            alive &= ~wm.touching(state, SPIKE)
            won = torch.where(alive & (won == 0) & wm.touching(state, EXIT), t + 1, won)
        h = wm.heuristic(state).tolist()
        imagined = [s[:2] for s, ok in zip(state.tolist(), alive.tolist()) if ok]
        for k, s in enumerate(state.tolist()):
            if not alive[k]:
                continue
            _, g, parent = batch[k // len(ACTIONS)]
            action = ACTIONS[k % len(ACTIONS)]
            if won[k]:
                nodes.append((parent, action, [tr[k] for tr in trace[:won[k]]]))
                return unroll(nodes, len(nodes) - 1), True, it + 1
            key = (round(s[0]), round(s[1]), round(s[2] * 2), round(s[3] * 2))
            if key in seen:
                continue
            seen.add(key)
            nodes.append((parent, action, [tr[k] for tr in trace]))
            heapq.heappush(frontier, (g + HOLD + 2 * h[k], g + HOLD, len(nodes) - 1))
            if h[k] < best_h:
                best, best_h = len(nodes) - 1, h[k]
        yield "search", start, (imagined, [st for st, _ in unroll(nodes, best)])
    return unroll(nodes, best), False, it + 1


def unroll(nodes, i):
    steps = []
    while nodes[i][0] is not None:
        parent, action, states = nodes[i]
        steps = [(s.tolist(), action) for s in states] + steps
        i = parent
    return steps


def play(level, wm, execute):
    """Plan with the model, act in the real engine, replan. Yields events:
    ("planning", real_state, None) before each plan, ("search", real_state, (imagined, best path)) as it
    searches, ("plan", real_state, predicted_path) once it has chosen,
    ("tick", real_state, (action, prediction_error, note)) for every real engine tick, ("done", state, message)."""
    state, deaths, replans = spawn_state(level), 0, 0
    queue, done_this_plan = [], 0
    for tick in range(MAX_TICKS):
        if not queue or done_this_plan >= execute:
            yield "planning", state, None
            t0 = time.time()
            queue, reached, iters = yield from plan(wm, state)
            replans += 1
            print(f"plan {replans} from ({state[0]:.1f}, {state[1]:.1f}): {len(queue)} ticks, "
                  f"{'reaches the exit' if reached else 'partial'}, {iters} search steps, {time.time() - t0:.1f}s")
            if not queue:
                yield "done", state, "the model can't find any move"
                return
            yield "plan", state, [s for s, _ in queue]
            done_this_plan = 0
        predicted, action = queue.pop(0)
        done_this_plan += 1
        ns, dead, state, finished = advance(state, action, level)
        err = max(abs(ns[0] - predicted[0]), abs(ns[1] - predicted[1]))
        note = None
        if dead:
            deaths += 1
            note = "died: replanning from spawn"
            queue = []
        elif err > REPLAN_DIST:
            note = f"reality drifted {err:.1f} px from the model: replanning"
            queue = []
        yield "tick", (state if not finished else ns), (action, err, note)
        if finished:
            yield "done", ns, f"finished in {tick + 1} ticks ({(tick + 1) / TICK_HZ:.1f}s), {replans} plans, {deaths} deaths"
            return
        if deaths == MAX_DEATHS:
            yield "done", state, f"gave up after {deaths} deaths"
            return
    yield "done", state, "gave up: out of time"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("level", nargs="?", default="level1")
    ap.add_argument("--model", default=MODEL_PATH, type=Path)
    ap.add_argument("--gif", type=Path, help="also record everything shown live to this GIF")
    ap.add_argument("--headless", action="store_true", help="no window, only write --gif")
    ap.add_argument("--execute", type=int, default=EXECUTE, help=f"real ticks per plan before replanning (default {EXECUTE})")
    args = ap.parse_args()

    level = load_level(args.level)
    wm = WorldModel(resolve_level(args.level).read_text(), args.model)
    if not (wm.grid == EXIT).any():
        raise SystemExit(f"{args.level} has no exit")

    if args.headless and not args.gif:
        raise SystemExit("--headless needs --gif")
    if args.headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
    import pygame

    pygame.init()
    size = (LEVEL_PX * SCALE, LEVEL_PX * SCALE)
    screen = pygame.display.set_mode(size)
    pygame.display.set_caption(f"platformNet - world model playing {args.level}")
    font = pygame.font.SysFont("consolas", 16, bold=True)
    clock = pygame.time.Clock()

    background = pygame.Surface(size)
    background.fill((0xBC, 0xDF, 0xFF))
    colors = {SOLID: (0, 0, 0), EXIT: (241, 196, 15), SPIKE: (231, 76, 60)}
    for (r, c), t in zip(torch.cartesian_prod(torch.arange(100), torch.arange(100)).tolist(), wm.grid.flatten().tolist()):
        if t in colors:
            pygame.draw.rect(background, colors[t], (c * TILE * SCALE, r * TILE * SCALE, TILE * SCALE, TILE * SCALE))

    def center(s):
        return ((s[0] + PLAYER / 2) * SCALE, (s[1] + PLAYER / 2) * SCALE)

    history = background.copy()
    scratch = pygame.Surface(size, pygame.SRCALPHA)

    def set_path(new):
        nonlocal path
        if len(path) > 1 and new != path:
            scratch.fill((0, 0, 0, 0))
            pygame.draw.lines(scratch, (220, 30, 30, 45), False, path, 2)
            history.blit(scratch, (0, 0))
        path = new

    def draw_player(surface, state, trail):
        if len(trail) > 1:
            pygame.draw.lines(surface, (255, 220, 0), False, trail, 2)
        pygame.draw.rect(surface, (230, 40, 160), (state[0] * SCALE, state[1] * SCALE, PLAYER * SCALE, PLAYER * SCALE))

    def draw_text(lines):
        for n, line in enumerate(lines):
            screen.blit(font.render(line, True, (0, 0, 0)), (10, 10 + 20 * n))

    def draw(state, lines):
        screen.blit(history, (0, 0))
        for p in explored:
            screen.blit(dot, p)
        if len(path) > 1:
            pygame.draw.lines(screen, (30, 190, 70), False, path, 3)
        draw_player(screen, state, trail)
        draw_text(lines)
        pygame.display.flip()
        nonlocal drawn
        drawn += 1
        if args.gif and (drawn % 2 or last):
            frames.append(pygame.transform.smoothscale(screen, (size[0] // 2, size[1] // 2)))

    dot = pygame.Surface((3, 3), pygame.SRCALPHA)
    dot.fill((40, 90, 230, 110))
    trail, explored, path, frames = [], [], [], []
    run = []
    drawn, last, quit_early = 0, False, False
    plans, err, status, keys = 0, 0.0, "", "     "
    for kind, state, info in play(level, wm, args.execute):
        if any(ev.type == pygame.QUIT for ev in pygame.event.get()):
            quit_early = True
            break
        if kind == "planning":
            trail.append(center(state))
            explored, searched, path = [], 0, []
            history.blit(background, (0, 0))
            continue
        if kind == "search":
            searched += 1
            explored += [(cx - 1, cy - 1) for cx, cy in map(center, info[0])]
            set_path([center(state)] + [center(s) for s in info[1]])
            draw(state, [f"keys [{keys}]", f"plans {plans}",
                         f"model is planning: {searched} steps, {len(explored)} futures imagined"])
            continue
        if kind == "plan":
            plans += 1
            explored = []
            set_path([center(state)] + [center(s) for s in info])
            continue
        if kind == "done":
            status, last = info, True
            print(info)
            draw(state, [f"keys [{keys}]", f"plans {plans}", status])
            run.append((state, keys, False))
            break
        (d, j), err, note = info
        if note:
            status = note
            print(note)
        died = bool(note and note.startswith("died"))
        if died:
            trail = []
        else:
            trail.append(center(state))
        keys = f"{'<' if d < 0 else ' '} {'^' if j else ' '} {'>' if d > 0 else ' '}"
        run.append((state, keys, died))
        draw(state, [f"keys [{keys}]", f"plans {plans}   model error {err:.2f} px", status])
        if not args.headless:
            clock.tick(TICK_HZ)

    if args.gif and frames:
        from PIL import Image
        frames += [frames[-1]] * 30
        imgs = [Image.frombytes("RGB", f.get_size(), pygame.image.tobytes(f, "RGB")) for f in frames]
        imgs[0].save(args.gif, save_all=True, append_images=imgs[1:], duration=1000 * 2 // TICK_HZ, loop=0)
        print(f"wrote {args.gif} ({len(imgs)} frames)")

    while not (args.headless or quit_early) and run:
        replay_trail = [center(spawn_state(level))]
        for state, keys, died in run + [run[-1]] * TICK_HZ:
            if any(ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE)
                   for ev in pygame.event.get()):
                pygame.quit()
                return
            replay_trail = [] if died else replay_trail + [center(state)]
            screen.blit(background, (0, 0))
            draw_player(screen, state, replay_trail)
            draw_text([f"keys [{keys}]", f"replay: {len(run)} ticks  (Esc to quit)"])
            pygame.display.flip()
            clock.tick(TICK_HZ)
    pygame.quit()

if __name__ == "__main__":
    main()
