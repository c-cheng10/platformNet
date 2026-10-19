# platformNet

A 2D platformer whose physics run inside a neural network.

We built a small, deterministic platformer engine, generated millions of examples of
"keys + current state → next state" from it, and trained a network to predict the next frame.
The neural game then runs on that network instead of the engine. Its mistakes (clipping
through walls, floating, sliding) are the glitches you get to hunt for.

## Quick start

You need Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bat
:: the neural game: physics come from the trained model
play.bat            :: NVIDIA GPU
play.bat cpu        :: any computer
```
On Mac/Linux use `./play.sh` (CPU) or `./play.sh gpu`.
Watch the model play a level by itself (it plans with the network, the real engine executes, then it replans):
`./demo.sh level2` opens a window; add `--gif demo.gif` to also save the run as a GIF.
The first run installs PyTorch and the other dependencies into `training/.venv`, which takes a few minutes.

**No uv?** Both scripts fall back to running `programs/render.py` with the first Python they find
that already has `torch`, `pygame` and `numpy` (Windows tries `py -3.12`, `py -3`, `python`, `python3`).
If there isn't one, they tell you what to install (`pip install torch pygame numpy`).

The real-engine tools only need pygame (`pip install pygame numpy`):

```bat
py -3.12 programs\game.py level2          :: play the real engine
py -3.12 programs\level_editor.py level4  :: draw a level (saves to levels\level4.txt)
```

## Folders

| Folder | What's in it |
|---|---|
| `programs/` | Things you run: `game.py` (real engine), `level_editor.py`, `render.py` (neural game, started by `play.bat`) |
| `game_py/` | The engine: `platformnet/` (physics, levels, spikes and exit) and its tests |
| `levels/` | Every level, as 100×100 ASCII files. All programs read and write here |
| `dataset-gen/` | Making training data: `gamerecorder.py` (record yourself playing), `generate_*.py` (generated data), `map_gen.py` (random maps) |
| `training/` | The model (`model.py`), `train.py`, and the Python environment (`pyproject.toml`, `uv.lock`) |
| `models/` | Trained model files (`.pt`). `programs/render.py` picks which one to use (`MODEL_PATH`) |
| `assets/` | `sprites/` for the real game, `art/` (player faces) and `art/bg/` (backgrounds) for the neural game |
| `java_game/` | The original Java version this started from |

## Levels

Levels are 100×100 characters, one per 5 px tile:

| Char | Tile |
|---|---|
| `.` | empty |
| `#` | solid |
| `X` | spike (kills you) |
| `E` | exit (finishes the level) |
| `S` | spawn (the player's top-left; exactly one) |

Every program takes a short name (`level2` means `levels/level2.txt`) or a full path.
The neural game plays `level1`, `level2`, … in order and stops at the first missing number.

**Level editor:** `1` solid, `2` spike, `3` exit, `4` spawn. Left-click paints. Right-click or Ctrl+click erases.
Shift+drag fills a rectangle. `[` `]` change the brush size, `G` toggles the grid, Ctrl+Z undoes,
**Ctrl+S saves**, Ctrl+N clears, Esc quits without saving.

**Design limits:** a jump is about 10 tiles high and clears gaps of up to about 18 tiles at full speed
(12–14 is comfortable). The player is 2×2 tiles.

## Physics (the ground truth)

Fixed 60 ticks/s. The state is only `x, y, vx, vy`. The next frame depends only on the keys,
that state and the level tiles: no timers, no randomness, no hidden state. That's what makes it
learnable.

- **Ground:** each material has a friction coefficient μ. `vx` moves toward ±3.3 px/tick by at most μ × 0.5 per tick.
- **Air:** air resistance removes 5% of `vx` per tick. Holding a direction balances it at 3.3.
- **Jump:** fixed height (about 55 px). **Gravity:** 0.6 px/tick². **Fall cap:** 15 px/tick.
- **Collision:** swept, so the player never tunnels through thin walls.
- **Spikes:** respawn you. **Exit:** finishes the level.

All constants are in `game_py/platformnet/physics.py`. Changing any of them makes existing data and models out of date.

## Data and training
https://huggingface.co/datasets/Kenzo-A-T/platformNetData is the final dataset used to train the model in use

```bat
cd dataset-gen
py -3.12 gamerecorder.py level2                     :: play and record → dataset-gen\recordings\
py -3.12 generate_data.py -n 250000                 :: generated pairs → dataset-gen\generated\
cd ..\training
uv run --extra gpu python train.py --latest         :: train on the newest generated file → models\model.pt
```
Each `generate_*.py` script explains its own mix at the top of the file (`--help` lists its options).

## Tests

```bat
cd game_py && py -3.12 -m pytest -q          :: physics
py -3.12 -m pytest -q dataset-gen\tests      :: map generator
```
