"""Train NextStatePredictor on recordings from game_py/gamerecorder.py.

    python train.py                          # every recording in ../game_py/recordings
    python train.py some/dir run1.json ...   # specific files / directories
    python train.py --epochs 50 --batch-size 32 --lr 3e-4 --out model.pt

Each recording is a list of {"input": {...}, "output": {...}} pairs, one per tick.

The network doesn't predict the raw next position/velocity: it predicts the *change* from the
previous frame (next - prev), standardized by the training set's mean/std. Positions are in
the hundreds of pixels but move a few pixels per tick, so predicting the change is far easier
to get precise. The stats are saved in the checkpoint; use load_predictor() to get real units back.
"""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

from model import LEVEL_SIZE, NextStatePredictor

RECORDINGS_DIR = Path(__file__).resolve().parent.parent / "game_py" / "recordings"

# Must match the tile codes in game_py/platformnet/physics.py (EMPTY, SOLID, EXIT, SPIKE = range(4)).
# Spawn ("S") is empty for physics.
CHAR_TO_TILE = {".": 0, "#": 1, "E": 2, "X": 3, "S": 0}

KEYS = ("shouldLeft", "shouldRight", "wantsJump")
PREV_STATE = ("prevposx", "prevposy", "prevvelx", "prevvely")
NEXT_STATE = ("posx", "posy", "velx", "vely")
FLAGS = ("isDead", "isFinished")


def parse_level(text):
    rows = [line for line in text.splitlines() if line.strip() and not line.startswith(";")]
    assert len(rows) == LEVEL_SIZE and all(len(r) == LEVEL_SIZE for r in rows), "level must be 100x100"
    return torch.tensor([[CHAR_TO_TILE[ch] for ch in row] for row in rows], dtype=torch.uint8)


def find_recordings(paths):
    files = []
    for p in map(Path, paths):
        files += sorted(p.glob("*.json")) if p.is_dir() else [p]
    return files


def load_recordings(files):
    """-> levels (N,100,100) uint8, keys (N,3), prev (N,4), next (N,4), flags (N,2)"""
    level_cache = {}  # every frame of a run repeats the same level text; parse it once
    levels, keys, prev, nxt, flags = [], [], [], [], []
    for f in files:
        data = json.loads(Path(f).read_text())
        # generate_data.py files are {"levels": [text, ...], "pairs": [...]}, LEVEL being an index into levels
        texts, pairs = (data["levels"], data["pairs"]) if isinstance(data, dict) else (None, data)
        for pair in pairs:
            i, o = pair["input"], pair["output"]
            text = texts[i["LEVEL"]] if texts is not None else i["LEVEL"]
            if text not in level_cache:
                level_cache[text] = parse_level(text)
            levels.append(level_cache[text])
            keys.append([i[k] for k in KEYS])
            prev.append([i[k] for k in PREV_STATE])
            nxt.append([o[k] for k in NEXT_STATE])
            flags.append([o[k] for k in FLAGS])
    as_float = lambda rows: torch.tensor(rows, dtype=torch.float32)
    return torch.stack(levels), as_float(keys), as_float(prev), as_float(nxt), as_float(flags)


def mean_std(x):
    return x.mean(0), x.std(0).clamp(min=1e-3)


def load_predictor(path="model.pt", device="cpu"):
    """Returns predict(level_text, keys, prev) -> (next_state (B,4) in pixels, flags (B,2) in 0-1).
    keys is (B,3) [left, right, jump], prev is (B,4) [x, y, vx, vy]."""
    ckpt = torch.load(path, map_location=device)
    model = NextStatePredictor(ckpt["hidden_size"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    s = {k: v.to(device) for k, v in ckpt["stats"].items()}

    @torch.no_grad()
    def predict(level_text, keys, prev):
        keys, prev = torch.as_tensor(keys, dtype=torch.float32, device=device), torch.as_tensor(prev, dtype=torch.float32, device=device)
        level = parse_level(level_text).to(device).expand(len(prev), -1, -1)
        extra = torch.cat([keys, (prev - s["in_mean"]) / s["in_std"]], dim=1)
        out = model(level, extra)
        return prev + out[:, :4] * s["delta_std"] + s["delta_mean"], out[:, 4:]

    return predict


_predictors = {}  # checkpoint path -> predict(), so the model is only loaded once


def run_model(inputs, path="model.pt"):
    """One forward pass with the trained parameters, in the recording format:
    inputs is a recording's "input" dict (shouldLeft, shouldRight, wantsJump, LEVEL, prevposx, ...),
    returns a dict like a recording's "output" (posx, posy, velx, vely, isDead, isFinished),
    with isDead / isFinished as probabilities in 0-1."""
    if path not in _predictors:
        _predictors[path] = load_predictor(path)
    state, flags = _predictors[path](inputs["LEVEL"], [[inputs[k] for k in KEYS]], [[inputs[k] for k in PREV_STATE]])
    values = state[0].tolist() + flags[0].tolist()
    return dict(zip(NEXT_STATE + FLAGS, values))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="*", default=[RECORDINGS_DIR], help="recording files or directories")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--out", default="model.pt")
    args = parser.parse_args()

    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    files = find_recordings(args.paths)
    levels, keys, prev, nxt, flags = load_recordings(files)
    print(f"{len(levels)} frames from {len(files)} recording(s), {len(torch.unique(levels, dim=0))} distinct level(s)")

    n_val = max(1, int(len(levels) * args.val_fraction))
    dataset = TensorDataset(levels, keys, prev, nxt, flags)
    train_set, val_set = random_split(dataset, [len(levels) - n_val, n_val])

    # Normalization stats come from the training split only, without death frames: a death teleports
    # the player back to spawn (hundreds of px), which would blow up the std and swamp the loss.
    # Deaths are masked out of the position/velocity loss and only train isDead.
    idx = torch.tensor(train_set.indices)
    idx = idx[flags[idx, 0] == 0]
    in_mean, in_std = mean_std(prev[idx])
    delta_mean, delta_std = mean_std((nxt - prev)[idx])
    stats = {"in_mean": in_mean, "in_std": in_std, "delta_mean": delta_mean, "delta_std": delta_std}
    stats_dev = {k: v.to(device) for k, v in stats.items()}

    def to_batch(batch):
        level, k, p, n, f = (t.to(device) for t in batch)
        extra = torch.cat([k, (p - stats_dev["in_mean"]) / stats_dev["in_std"]], dim=1)
        target = torch.cat([((n - p) - stats_dev["delta_mean"]) / stats_dev["delta_std"], f], dim=1)
        alive = 1 - f[:, 0]
        return level, extra, target, p, n, alive

    # drop_last: BatchNorm can't train on a batch of one
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=len(train_set) > args.batch_size)
    val_loader = DataLoader(val_set, batch_size=256)

    model = NextStatePredictor(args.hidden_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = seen = 0
        for batch in train_loader:
            level, extra, target, _, _, alive = to_batch(batch)
            loss = model.loss(level, extra, target, alive)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(level)
            seen += len(level)
        train_loss /= seen
        scheduler.step()

        # Validation, also reported in real units: pixel error of position and flag accuracy
        model.eval()
        val_loss = pos_err = vel_err = flag_correct = n_alive = 0.0
        with torch.no_grad():
            for batch in val_loader:
                level, extra, target, p, n, alive = to_batch(batch)
                val_loss += model.loss(level, extra, target, alive).item() * len(level)
                out = model(level, extra)
                pred = p + out[:, :4] * stats_dev["delta_std"] + stats_dev["delta_mean"]
                pos_err += ((pred[:, :2] - n[:, :2]).norm(dim=1) * alive).sum().item()
                vel_err += ((pred[:, 2:] - n[:, 2:]).abs().mean(1) * alive).sum().item()
                flag_correct += ((out[:, 4:] > 0.5).float() == target[:, 4:]).all(1).sum().item()
                n_alive += alive.sum().item()
        val_loss, flag_acc = val_loss / n_val, flag_correct / n_val
        pos_err, vel_err = pos_err / max(n_alive, 1), vel_err / max(n_alive, 1)

        print(f"epoch {epoch:3d}  train {train_loss:.4f}  val {val_loss:.4f}  "
              f"pos err {pos_err:.2f}px  vel err {vel_err:.3f}px/tick  flags acc {flag_acc:.1%}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model": model.state_dict(), "stats": stats, "hidden_size": args.hidden_size}, args.out)

    print(f"best val loss {best_val:.4f}, saved to {args.out}")


if __name__ == "__main__":
    main()
