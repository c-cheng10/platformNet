# PROGRESS

## What works (milestone 1: real engine + frame logger)
- `game_py/`: Python port of the Java game's physics (`game/Game.java`), same constants and feel
  (accel 0.7, friction ×0.85, gravity 0.3, jump −8, hold-to-bunny-hop, 100 ticks/s), with fixes for the neural net:
  - swept collision that snaps to contact (Java undid the move → history-dependent resting positions)
  - fall speed capped at 10 px/tick (Java's was unbounded)
  - hitbox = sprite (Java drew a 40 px sprite over a 5 px hitbox), touching an edge ≠ colliding
  - explicit `grounded` / `on_ice` state; teleports (death, room change) flagged per tick in `events`
- Pure `step(state, action, level)` in `platformnet/physics.py`, no Pygame. ~108k ticks/s headless.
- Tiles: solid, ice, bounce, one-way, spike, exit. Six single-screen rooms (500×500 px, 5 px tiles) in `levels/zone*.txt`.
- `python game.py` → keyboard play. Every tick is logged to `data/human/ep_NNNNNN.npz`
  (states [T,6], actions [T,2], next_states [T,6], episode_ids, zone_ids, events, start state, physics/level hashes).
- Determinism: replaying any episode's actions reproduces states bit-for-bit (`tests/test_determinism.py`,
  also re-checks every recorded episode on disk). 17 tests pass.
- Search bot clears all six rooms without clipping (≈3 s each), so the level is beatable.

## Run
```
cd game_py
pip install -r requirements.txt
python game.py            # --zone N to start in room N, --no-log to not record
python -m pytest -q
```
Keys: arrows/WASD, Up/Space/W jump, R restart room, N/P next/prev room, Esc quit.

## Next
1. Human playtest: tune feel (ICE_FRICTION, BOUNCE_VEL, MAX_FALL) and room layouts **before** recording real data.
2. Headless bots: random + scripted per room; data generator with exclusion rules (e.g. no ice→bounce jumps).
3. MLP world model on stacked frames predicting Δstate, with zone_id as an input.

## Known issues / notes
- State is 6-D, not the 7-D in the original brief: the Java jump is hold-triggered, so no `jump_held` is needed.
- Rooms are short and easy (bot: ~20 s full run). Fine for now; iterate after playtest.
- Changing any physics constant invalidates recorded data: bump `PHYSICS_VERSION` and re-record.
- `levels/*.txt` are the source of truth; `tools/make_levels.py` overwrites them (one-off sketch).
