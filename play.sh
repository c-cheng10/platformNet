#!/usr/bin/env sh
cd "$(dirname "$0")" || exit 1

if command -v uv >/dev/null 2>&1; then
    exec uv run --project training --extra "${1:-cpu}" python programs/render.py
fi

for py in python3 python; do
    if command -v "$py" >/dev/null 2>&1 && "$py" -c "import torch, pygame, numpy" >/dev/null 2>&1; then
        echo "uv not found, running with $py instead."
        exec "$py" programs/render.py
    fi
done

echo "uv is not installed, and no Python with torch, pygame and numpy was found." >&2
echo "Either install uv: https://docs.astral.sh/uv/getting-started/installation/" >&2
echo "or install the packages:  pip install torch pygame numpy" >&2
exit 1
