"""Train NextStatePredictor on recordings from game_py/gamerecorder.py.

    python train.py                          # every recording in ../game_py/recordings
    python train.py some/dir run1.json ...   # specific files / directories
    python train.py --epochs 50 --batch-size 32 --lr 3e-4 --out model.pt
    python train.py --latest                 # newest dataset in ../game_py/generated

Each recording is a list of {"input": {...}, "output": {...}} pairs, one per tick.

The network doesn't predict the raw next position/velocity: it predicts the *change* from the
previous frame (next - prev), standardized by the training set's mean/std. Positions are in
the hundreds of pixels but move a few pixels per tick, so predicting the change is far easier
to get precise. The stats are saved in the checkpoint; use load_predictor() to get real units back.

The network only sees a 32x32-tile crop of the level around the player (the tiles under the player's
top-left corner at row/col 15), plus the keys, where the player sits inside that tile (frac x/y, 0-1)
and its velocity. Absolute position isn't an input: the physics is the same everywhere in the level.
Cropping happens once at load time.

Death frames are skipped: a death teleports the player back to spawn (hundreds of px), which isn't
physics the network should learn and would blow up the std and swamp the loss.
"""
import argparse
import gzip
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split

from model import CROP, HALF, LEVEL_SIZE, NextStatePredictor

RECORDINGS_DIR = Path(__file__).resolve().parent.parent / "game_py" / "recordings"
GENERATED_DIR = Path(__file__).resolve().parent.parent / "game_py" / "generated"

# Must match the tile codes in game_py/platformnet/physics.py (EMPTY, SOLID, EXIT, SPIKE = range(4)).
# Spawn ("S") is empty for physics.
CHAR_TO_TILE = {".": 0, "#": 1, "E": 2, "X": 3, "S": 0}
SOLID = 1
TILE = 5  # px per tile
PAD = CROP  # solid border added around the level, so crops near (or past) the edge stay in bounds

KEYS = ("shouldLeft", "shouldRight", "wantsJump")
PREV_STATE = ("prevposx", "prevposy", "prevvelx", "prevvely")
NEXT_STATE = ("posx", "posy", "velx", "vely")


def parse_level(text):
    rows = [line for line in text.splitlines() if line.strip() and not line.startswith(";")]
    assert len(rows) == LEVEL_SIZE and all(len(r) == LEVEL_SIZE for r in rows), "level must be 100x100"
    return torch.tensor([[CHAR_TO_TILE[ch] for ch in row] for row in rows], dtype=torch.uint8)


def pad_level(level):
    """(100,100) -> (100+2*PAD, 100+2*PAD), surrounded by solid like the game treats out-of-bounds"""
    padded = torch.full((LEVEL_SIZE + 2 * PAD, LEVEL_SIZE + 2 * PAD), SOLID, dtype=torch.uint8)
    padded[PAD:PAD + LEVEL_SIZE, PAD:PAD + LEVEL_SIZE] = level
    return padded


def frame_inputs(padded, keys, prev):
    """One frame -> crop (32,32) and extra [left, right, jump, frac_x, frac_y, vx, vy] (unnormalized)"""
    x, y, vx, vy = prev
    col, row = int(x // TILE) + PAD - HALF, int(y // TILE) + PAD - HALF
    crop = padded[row:row + CROP, col:col + CROP]
    return crop, list(keys) + [(x % TILE) / TILE, (y % TILE) / TILE, vx, vy]


def find_recordings(paths):
    """Files and directories -> data files. Directories contribute every .json and .json.gz,
    skipping a .json.gz whose unzipped .json sits next to it so nothing loads twice."""
    files = []
    for p in map(Path, paths):
        if not p.is_dir():
            files.append(p)
            continue
        jsons = sorted(p.glob("*.json"))
        files += jsons + [gz for gz in sorted(p.glob("*.json.gz")) if gz.with_suffix("") not in jsons]
    return files


def latest_generated():
    """The most recently written dataset in game_py/generated (generate_data.py / generate_clumped_data.py)"""
    files = list(GENERATED_DIR.glob("*.json")) + list(GENERATED_DIR.glob("*.json.gz"))
    if not files:
        raise SystemExit(f"no datasets in {GENERATED_DIR}")
    return max(files, key=lambda f: f.stat().st_mtime)


def read_pairs(path):
    """Yields (level_text, pair) from either data format, plain or gzipped:
    a list of pairs whose LEVEL is the level text (gamerecorder.py), or
    {"levels": [text, ...], "pairs": [...]} whose LEVEL is an index into levels (generate_data.py)."""
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as f:
        data = json.load(f)
    if isinstance(data, dict):
        levels = data["levels"]
        for pair in data["pairs"]:
            yield levels[pair["input"]["LEVEL"]], pair
    else:
        for pair in data:
            yield pair["input"]["LEVEL"], pair


def load_recordings(files):
    """-> crops (N,32,32) uint8, extra (N,7), prev (N,4), next (N,4), number of distinct levels.
    Death frames are skipped."""
    level_cache = {}  # many frames share a level; parse and pad each one once
    crops, extra, prev, nxt = [], [], [], []
    for f in files:
        for level_text, pair in read_pairs(f):
            i, o = pair["input"], pair["output"]
            if o["isDead"]:
                continue
            if level_text not in level_cache:
                level_cache[level_text] = pad_level(parse_level(level_text))
            p = [i[k] for k in PREV_STATE]
            crop, e = frame_inputs(level_cache[level_text], [i[k] for k in KEYS], p)
            crops.append(crop)
            extra.append(e)
            prev.append(p)
            nxt.append([o[k] for k in NEXT_STATE])
    as_float = lambda rows: torch.tensor(rows, dtype=torch.float32)
    return torch.stack(crops), as_float(extra), as_float(prev), as_float(nxt), len(level_cache)


def normalize_extra(extra, s):
    # keys and frac are already 0-1; only the velocity needs standardizing
    return torch.cat([extra[:, :5], (extra[:, 5:] - s["vel_mean"]) / s["vel_std"]], dim=1)


def mean_std(x):
    return x.mean(0), x.std(0).clamp(min=1e-3)


def load_predictor(path="model.pt", device="cpu"):
    """Returns predict(level_text, keys, prev) -> next_state (B,4) [x, y, vx, vy] in pixels.
    keys is (B,3) [left, right, jump], prev is (B,4) [x, y, vx, vy]."""
    ckpt = torch.load(path, map_location=device)
    model = NextStatePredictor(ckpt["hidden_size"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    s = {k: v.to(device) for k, v in ckpt["stats"].items()}
    padded_cache = {}

    @torch.no_grad()
    def predict(level_text, keys, prev):
        if level_text not in padded_cache:
            padded_cache[level_text] = pad_level(parse_level(level_text))
        frames = [frame_inputs(padded_cache[level_text], k, p) for k, p in zip(keys, prev)]
        crops = torch.stack([c for c, _ in frames]).to(device)
        extra = normalize_extra(torch.tensor([e for _, e in frames], dtype=torch.float32, device=device), s)
        prev = torch.as_tensor(prev, dtype=torch.float32, device=device)
        return prev + model(crops, extra) * s["delta_std"] + s["delta_mean"]

    return predict


_predictors = {}  # checkpoint path -> predict(), so the model is only loaded once


def run_model(inputs, path="model.pt"):
    """One forward pass with the trained parameters, in the recording format:
    inputs is a recording's "input" dict (shouldLeft, shouldRight, wantsJump, LEVEL, prevposx, ...),
    returns a dict of the next state (posx, posy, velx, vely)."""
    if path not in _predictors:
        _predictors[path] = load_predictor(path)
    state = _predictors[path](inputs["LEVEL"], [[inputs[k] for k in KEYS]], [[inputs[k] for k in PREV_STATE]])
    return dict(zip(NEXT_STATE, state[0].tolist()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="*", default=[RECORDINGS_DIR], help="recording files or directories")
    parser.add_argument("--latest", action="store_true", help="train on the newest dataset in game_py/generated")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--out", default="model.pt")
    parser.add_argument("--patience", type=int, default=10,
                        help="stop after this many epochs without val loss improving by --min-delta (0 = never stop early)")
    parser.add_argument("--min-delta", type=float, default=1e-4, help="smallest val loss drop that counts as improving")
    args = parser.parse_args()

    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def sync():
        # CUDA runs asynchronously; wait for it so the timings land in the right bucket
        if device == "cuda":
            torch.cuda.synchronize()

    load_start = time.perf_counter()
    files = [latest_generated()] if args.latest else find_recordings(args.paths)
    crops, extra, prev, nxt, n_levels = load_recordings(files)
    print(f"{len(crops)} frames from {len(files)} recording(s), {n_levels} distinct level(s), "
          f"loaded in {time.perf_counter() - load_start:.1f}s")

    n_val = max(1, int(len(crops) * args.val_fraction))
    dataset = TensorDataset(crops, extra, prev, nxt)
    train_set, val_set = random_split(dataset, [len(crops) - n_val, n_val])

    # Normalization stats come from the training split only
    idx = torch.tensor(train_set.indices)
    vel_mean, vel_std = mean_std(prev[idx, 2:])
    delta_mean, delta_std = mean_std((nxt - prev)[idx])
    stats = {"vel_mean": vel_mean, "vel_std": vel_std, "delta_mean": delta_mean, "delta_std": delta_std}
    stats_dev = {k: v.to(device) for k, v in stats.items()}

    def to_batch(batch):
        level, e, p, n = (t.to(device) for t in batch)
        extra = normalize_extra(e, stats_dev)
        target = ((n - p) - stats_dev["delta_mean"]) / stats_dev["delta_std"]
        return level, extra, target, p, n

    # drop_last: BatchNorm can't train on a batch of one
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=len(train_set) > args.batch_size)
    val_loader = DataLoader(val_set, batch_size=256)

    model = NextStatePredictor(args.hidden_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    best_val, best_epoch = float("inf"), 0
    total_data = total_train = total_eval = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = seen = 0
        data_time = train_time = eval_time = 0.0
        t = time.perf_counter()
        for batch in train_loader:
            # data: fetching the batch from the DataLoader + normalizing / moving it to the device
            level, extra, target, _, _ = to_batch(batch)
            sync()
            data_time += time.perf_counter() - t
            t = time.perf_counter()

            # train: forward, backward, optimizer step
            loss = model.loss(level, extra, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(level)  # .item() also waits for the GPU
            seen += len(level)
            train_time += time.perf_counter() - t
            t = time.perf_counter()
        train_loss /= seen
        scheduler.step()

        # Validation, also reported in real units: mean absolute error of x, y (px) and vx, vy (px/tick),
        # plus the straight-line position error
        model.eval()
        val_loss = dist_err = 0.0
        abs_err = torch.zeros(4, device=device)
        with torch.no_grad():
            t = time.perf_counter()
            for batch in val_loader:
                # data, same as in training
                level, extra, target, p, n = to_batch(batch)
                sync()
                data_time += time.perf_counter() - t
                t = time.perf_counter()

                # eval: forward pass + metrics
                out = model(level, extra)
                val_loss += F.mse_loss(out, target).item() * len(level)
                pred = p + out * stats_dev["delta_std"] + stats_dev["delta_mean"]
                abs_err += (pred - n).abs().sum(0)
                dist_err += (pred[:, :2] - n[:, :2]).norm(dim=1).sum().item()
                eval_time += time.perf_counter() - t
                t = time.perf_counter()
        val_loss, dist_err = val_loss / n_val, dist_err / n_val
        err_x, err_y, err_vx, err_vy = (abs_err / n_val).tolist()

        total_data, total_train, total_eval = total_data + data_time, total_train + train_time, total_eval + eval_time
        print(f"epoch {epoch:3d}  loss train {train_loss:.4f} val {val_loss:.4f} | "
              f"val err x {err_x:.2f} y {err_y:.2f} (dist {dist_err:.2f}) px, vx {err_vx:.3f} vy {err_vy:.3f} px/tick | "
              f"time data {data_time:.1f}s train {train_time:.1f}s eval {eval_time:.1f}s")

        if val_loss < best_val - args.min_delta:
            best_val, best_epoch = val_loss, epoch
            torch.save({"model": model.state_dict(), "stats": stats, "hidden_size": args.hidden_size}, args.out)
        elif args.patience and epoch - best_epoch >= args.patience:
            print(f"early stop: val loss hasn't improved for {args.patience} epochs")
            break

    total = total_data + total_train + total_eval
    print(f"time: data {total_data:.1f}s ({total_data / total:.0%})  train {total_train:.1f}s ({total_train / total:.0%})  "
          f"eval {total_eval:.1f}s ({total_eval / total:.0%})")
    print(f"best val loss {best_val:.4f} (epoch {best_epoch}), saved to {args.out}")


if __name__ == "__main__":
    main()
