@echo off
rem Play the neural game (programs\render.py): the physics runs inside the trained model.
rem   play.bat        GPU build of PyTorch (NVIDIA + CUDA)
rem   play.bat cpu    CPU-only PyTorch (any computer)
rem With uv (https://docs.astral.sh/uv/) the first run installs everything into training\.venv.
rem Without uv it falls back to the first Python that already has torch, pygame and numpy.
cd /d "%~dp0"

where uv >nul 2>nul
if errorlevel 1 goto :no_uv
set EXTRA=gpu
if /i "%~1"=="cpu" set EXTRA=cpu
uv run --project training --extra %EXTRA% python programs\render.py
exit /b %errorlevel%

:no_uv
for %%P in ("py -3.12" "py -3" "python" "python3") do (
    %%~P -c "import torch, pygame, numpy" >nul 2>nul && (
        set "PYEXE=%%~P"
        goto :run_python
    )
)
echo uv is not installed, and no Python with torch, pygame and numpy was found.
echo Either install uv: https://docs.astral.sh/uv/getting-started/installation/
echo or install the packages:  pip install torch pygame numpy
exit /b 1

:run_python
echo uv not found, running with "%PYEXE%" instead.
%PYEXE% programs\render.py
exit /b %errorlevel%
