import json
from pathlib import Path
import numpy as np
from platformnet.logger import DATA_DIR
from platformnet.level import LEVEL_PATH

def export_human_data_to_json(output_filename="inputs_dataset.json"):
    human_dir = DATA_DIR / "human"
    if not human_dir.exists():
        print(f"Directory {human_dir} not found. Play some games first or disable --no-log.")
        return

    # Load the actual raw text contents of the level file (e.g., level1.txt)
    level_content = LEVEL_PATH.read_text() if LEVEL_PATH.exists() else ""

    dataset = []

    # Iterate through all recorded numpy archive episode files
    for file_path in human_dir.glob("*.npz"):
        data = np.load(file_path, allow_pickle=True)
        
        actions = data.get("actions")  # shape (T, 2): [input_dir, jump][cite: 1, 3]
        states = data.get("states")    # shape (T, 4): [x, y, vx, vy][cite: 3, 4]

        if actions is None or states is None:
            continue

        for t in range(len(actions)):
            input_dir, jump = int(actions[t][0]), int(actions[t][1])
            x, y, vx, vy = float(states[t][0]), float(states[t][1]), float(states[t][2]), float(states[t][3])

            should_left = 1 if input_dir < 0 else 0
            should_right = 1 if input_dir > 0 else 0
            wants_jump = int(jump)

            record = {
                "shouldleft": should_left,
                "shouldright": should_right,
                "wantsjump": wants_jump,
                "LEVEL": level_content,
                "prevposx": x,
                "prevposy": y,
                "prevvelx": vx,
                "prevvely": vy
            }
            dataset.append(record)

    output_path = DATA_DIR / output_filename
    output_path.write_text(json.dumps(dataset, indent=2))
    print(f"Successfully exported {len(dataset)} frames to {output_path.resolve()}")

if __name__ == "__main__":
    export_human_data_to_json()