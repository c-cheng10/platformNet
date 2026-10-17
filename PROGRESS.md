# PROGRESS

## What works
- `game_py/`: a minimal platformer engine derived from the Java game (`game/`), fixed 60 ticks/s.
  - Ground: each material has a friction coefficient μ (`FRICTION` in `physics.py`, solid = 1.0);
    vx moves toward ±3.3 px/tick by at most μ × 0.5 per tick (speeding up and slowing down).
  - Air: air resistance removes 5% of vx per tick; holding a direction balances it at 3.3.
  - Fixed-height jump (apex ~55 px), gravity 0.6, fall cap 15 px/tick, swept collision (no tunnelling).
  - Spikes kill and respawn you; the green exit finishes the level.
  - State is just x, y, vx, vy: the next frame depends only on keys, state and tiles (no hidden state).
- One level, `levels/level1.txt` (500×500 px, 5 px tiles).

## Run
```
cd game_py
pip install -r requirements.txt
python game.py                 # or: python game.py levels/other.txt
python level_editor.py         # draw levels; Ctrl+S saves ASCII to levels/level1.txt (or a path you pass)
python -m pytest -q
```
Keys: arrows/WASD, Up/Space/W jump, R restart, Esc quit.

## Layout
- `platformnet/physics.py`: `step(state, action, level)`, all the physics, no Pygame
- `platformnet/level.py`: loads the ASCII level (`.` empty, `#` solid, `X` spike, `E` exit, `S` spawn)
- `platformnet/run.py`: spikes and exit on top of the physics
- `game.py`: Pygame window, keyboard, drawing
- `level_editor.py`: mouse level editor. 1 solid, 2 spike, 3 exit, 4 spawn; left paint, right erase,
  Shift+drag rectangle, [ ] brush size, G grid, Ctrl+Z undo, Ctrl+S save (validated), Ctrl+N clear
