#!/usr/bin/env sh
cd "$(dirname "$0")" || exit 1
exec uv run --project training --extra cpu --with pillow python programs/demo.py "$@"
