@echo off
rem Play the neural game (programs\render.py): the physics runs inside the trained model.
rem   play.bat        GPU build of PyTorch (NVIDIA + CUDA)
rem   play.bat cpu    CPU-only PyTorch (any computer)
rem Needs uv (https://docs.astral.sh/uv/). The first run installs the dependencies into training\.venv.
cd /d "%~dp0"
set EXTRA=gpu
if /i "%~1"=="cpu" set EXTRA=cpu
uv run --project training --extra %EXTRA% python programs\render.py
