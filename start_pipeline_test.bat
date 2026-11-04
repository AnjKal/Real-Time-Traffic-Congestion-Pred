@echo off
rem Starts the pipeline test in two windows:
rem   1. pipeline: loads the model, polls TomTom every 2 minutes (24 polls = 2,400 calls, ~48 min),
rem      refreshes the forecast for all 100 junctions after every poll, and lets you TYPE a
rem      junction name or ID to see its current forecast.   (Python 3.7 + TensorFlow env)
rem   2. web page at http://127.0.0.1:8010/                  (your normal Python)
rem PIPELINE TEST: the model was trained on 15-minute readings; this checks the plumbing,
rem not the model's accuracy.
setlocal
set "HERE=%~dp0"
set "PY37=%HERE%..\.venv37\Scripts\python.exe"
if not exist "%PY37%" (
  echo Cannot find %PY37%
  echo Create the Python 3.7 environment first.
  pause
  exit /b 1
)
cd /d "%HERE%"
start "pipeline - type a junction name here" cmd /k ""%PY37%" start_pipeline.py --step-min 2"
start "web page" cmd /k "python run_webapp.py"
echo.
echo Open http://127.0.0.1:8010/ once the first window says "Model ready".
echo When the pipeline window reports 24 polls, check accuracy with:
echo     "%PY37%" evaluate_live.py --step-min 2
pause
