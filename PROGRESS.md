# PROGRESS

## What works (milestone 1: real engine + frame logger)
- `game_py/`: minimal engine derived from the Java game: acceleration, air/ground friction, fixed-height jump,
  gravity, solid collision, spikes. Fixed 60 ticks/s.
- **Markov by construction:** each frame's targets (next x, y, vx, vy, dead) are a pure function of exactly the
  network inputs (keys pressed, x, y, vx, vy, level tiles). No stored `grounded` (derived from position + tiles,
  since collisions snap to exact contact), no timers, no randomness. Tested by recomputing every logged frame
  in shuffled order from a level rebuilt from the logged tiles alone.
- Movement: on the ground, each material has a friction coefficient μ (`FRICTION` in physics.py, solid = 1.0); vx moves toward ±3.3 px/tick by at most μ × 0.5 per tick (speeding up and slowing down). In the air, air resistance removes 5% of vx per tick; holding a direction balances it at 3.3.
  Jump apex ~55 px (gravity 0.6, jump −8.4), fixed. Fall cap 15 px/tick, swept collision (no tunnelling).
- Pure `step(state, action, level)` in `platformnet/physics.py`, no Pygame.
- One level, `levels/level1.txt` (500×500 px, 5 px tiles): step, spike pit, platform, thin wall, spikes, exit.
- `python game.py` → keyboard play, no timer/death HUD. Every tick logged to `data/human/ep_NNNNNN.npz`:
  inputs `actions [T,2]`, `states [T,4]`, `tiles [100,100]`; targets `next_states [T,4]`, `dead [T]`.
- Replay reproduces logs bit-for-bit. 20 tests pass. Search bot clears the level (~3 s) without clipping.

## Run
```
cd game_py
pip install -r requirements.txt
python game.py            # --no-log to not record
python -m pytest -q
```
Keys: arrows/WASD, Up/Space/W jump, R restart, Esc quit.

## Next
1. Human playtest: confirm feel matches the Java game before recording real data.
2. Headless bots (random + scripted) and a data generator with exclusion rules.
3. MLP world model on stacked frames predicting Δstate.

## Known issues / notes
- Ice, bounce pads, one-way platforms and multiple rooms were cut on purpose to keep it simple. Add back later if time allows.
- Feeding all 10,000 tiles to the net is fine but they are constant with one level, so they carry no information until there are more levels.
- Changing any physics constant invalidates recorded data: bump `PHYSICS_VERSION` and re-record.
- `levels/level1.txt` is the source of truth; `tools/make_levels.py` overwrites it.
