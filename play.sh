#!/usr/bin/env sh
# Play the neural game (programs/render.py): the physics runs inside the trained model.
#   ./play.sh        CPU-only PyTorch (any computer, including Macs)
#   ./play.sh gpu    GPU build of PyTorch (NVIDIA + CUDA)
# Needs uv (https://docs.astral.sh/uv/). The first run installs the dependencies into training/.venv.
cd "$(dirname "$0")" || exit 1
exec uv run --project training --extra "${1:-cpu}" python programs/render.py
