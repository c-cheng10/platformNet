import json
import sys
import time
from pathlib import Path

# platformnet (the engine) and game.py live in <repo>/game_py
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "game_py"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "programs"))  # game.py
from platformnet.level import LEVEL_PATH, resolve_level  # noqa: E402

# Folder where game recordings are saved
RECORDINGS_DIR = Path(__file__).resolve().parent / "recordings"


def make_pair(level, state, action, next_state, dead, finished):
    """One training pair: what you did and what happened next. The generate_*.py scripts use this too."""
    x, y, vx, vy = state
    nx, ny, nvx, nvy = next_state

    return {
        "input": {
            "shouldLeft": 1 if action[0] < 0 else 0,
            "shouldRight": 1 if action[0] > 0 else 0,
            "wantsJump": int(action[1]),
            "LEVEL": level,
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
        # Load the level design
        self.level_text = resolve_level(level_path).read_text()
        self.out_dir = Path(out_dir)
        self.pairs = []

    def record(self, state, action, next_state, dead, finished):
        """Save what you did and what happened next as a pair."""
        self.pairs.append(make_pair(self.level_text, state, action, next_state, dead, finished))

    def end_episode(self):
        if not self.pairs:
            return None
        
        self.out_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{time.strftime('%Y%m%d-%H%M%S')}-{time.time_ns() % 10**9:09d}.json"
        path = self.out_dir / filename
        
        path.write_text(json.dumps(self.pairs, indent=2))
        print(f"Recorded {len(self.pairs)} pairs to {path}")
        
        self.pairs = []
        return path


if __name__ == "__main__":
    import game

    level_path = sys.argv[1] if len(sys.argv) > 1 else LEVEL_PATH
    game.main(level_path, GameRecorder(level_path))