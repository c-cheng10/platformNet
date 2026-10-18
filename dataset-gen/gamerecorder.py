"""Record human play as JSON input -> output pairs, one file per run.

    python gamerecorder.py [level]   play (same controls as game.py) and record

Every frame becomes one pair. The input is frame 1: the keys pressed and the state the player was in.
The output is frame 2: the state the player is actually in on the next frame (the spawn point after
a death), plus isDead / isFinished, which belong to frame 1 (did this frame's move hit a spike or
reach the exit). So each pair's output position/velocity equals the next pair's input.

Each run (spawn -> finish, restart or quit) is written to recordings/<timestamp>.json as a list of
{"input": {...}, "output": {...}} pairs.
"""
import json
import sys
import time
from pathlib import Path

# platformnet (the engine) and game.py live in <repo>/game_py
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "game_py"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "programs"))  # game.py
from platformnet.level import LEVEL_PATH, resolve_level  # noqa: E402

RECORDINGS_DIR = Path(__file__).resolve().parent / "recordings"


def make_pair(level_text, state, action, next_state, dead, finished):
    x, y, vx, vy = state
    nx, ny, nvx, nvy = next_state
    return {
        "input": {
            "shouldLeft": 1 if action[0] < 0 else 0,
            "shouldRight": 1 if action[0] > 0 else 0,
            "wantsJump": int(action[1]),
            "LEVEL": level_text,
            "prevposx": float(x),
            "prevposy": float(y),
            "prevvelx": float(vx),
            "prevvely": float(vy),
        },
        "output": {
            "posx": float(nx),
            "posy": float(ny),
            "velx": float(nvx),
            "vely": float(nvy),
            "isDead": int(dead),
            "isFinished": int(finished),
        },
    }


class GameRecorder:
    def __init__(self, level_path=LEVEL_PATH, out_dir=RECORDINGS_DIR):
        self.level_text = resolve_level(level_path).read_text()
        self.out_dir = Path(out_dir)
        self.pairs = []

    def record(self, state, action, next_state, dead, finished):
        """Log one frame: state + action, the next frame's state, and this frame's dead/finished."""
        self.pairs.append(make_pair(self.level_text, state, action, next_state, dead, finished))

    def end_episode(self):
        """Write the current run to its own file (if it has any frames) and start a new one."""
        if not self.pairs:
            return None
        self.out_dir.mkdir(parents=True, exist_ok=True)
        path = self.out_dir / f"{time.strftime('%Y%m%d-%H%M%S')}-{time.time_ns() % 10**9:09d}.json"
        path.write_text(json.dumps(self.pairs, indent=2))
        print(f"Recorded {len(self.pairs)} pairs to {path}")
        self.pairs = []
        return path


if __name__ == "__main__":
    import game

    level_path = sys.argv[1] if len(sys.argv) > 1 else LEVEL_PATH
    game.main(level_path, GameRecorder(level_path))
