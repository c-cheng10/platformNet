# PROGRESS

## What works (milestone 1: real engine + frame logger)
- `game_py/`: Python port of the Java game's physics (`game/Game.java`). Kept deliberately minimal:
  acceleration, friction, fixed-height jump, gravity/falling, solid collision, and a kill block (spike).
  Based on Java (friction ×0.85, gravity 0.3, hold-to-hop, 100 ticks/s) but slower and lower: accel 0.35 (Java 0.7), jump −6 (Java −8). Fixes for the neural net:
  - swept collision that snaps to contact (Java undid the move, so resting positions depended on history)
  - fall speed capped at 10 px/tick (Java's was unbounded)
  - hitbox = sprite (Java drew a 40 px sprite over a 5 px hitbox); touching an edge ≠ colliding
  - explicit `grounded` state; spike respawns flagged per tick in `events`
- No timer or death counter on screen (removed on purpose).
- Pure `step(state, action, level)` in `platformnet/physics.py`, no Pygame. ~100k ticks/s headless.
- One level, `levels/level1.txt` (500×500 px, 5 px tiles): step, spike pit, platform, thin wall, spikes, exit.
- `python game.py` → keyboard play. Every tick is logged to `data/human/ep_NNNNNN.npz`
  (states [T,5] = x, y, vx, vy, grounded; actions [T,2]; next_states [T,5]; episode_ids; events; start_state; physics/level hashes).
- Determinism: replaying any episode's actions reproduces states bit-for-bit (`tests/test_determinism.py`,
  which also re-checks every recorded episode on disk). 15 tests pass.
- A search bot clears the level in 3.5 s without clipping, so it is beatable.

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
- Changing any physics constant invalidates recorded data: bump `PHYSICS_VERSION` and re-record.
- `levels/level1.txt` is the source of truth; `tools/make_levels.py` overwrites it.
