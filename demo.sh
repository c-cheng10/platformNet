#!/usr/bin/env sh
# Watch the world model play a level by itself (programs/demo.py). Same setup as play.sh.
#   ./demo.sh level1                  live window
#   ./demo.sh level3 --gif out.gif    render to a GIF
cd "$(dirname "$0")" || exit 1
exec uv run --project training --extra cpu --with pillow python programs/demo.py "$@"
