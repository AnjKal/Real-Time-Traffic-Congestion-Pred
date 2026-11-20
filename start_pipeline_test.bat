@echo off
setlocal
set "HERE=%~dp0"
cd /d "%HERE%"

docker compose up --build -d
if errorlevel 1 (
  echo Docker Compose could not start the traffic pipeline.
  echo Check that Docker Desktop is running and that .env contains TOMTOM_API_KEY.
  pause
  exit /b 1
)

echo.
echo Kafka, the pipeline, and the map are running in the background.
echo Dashboard: http://localhost:8010/
echo Logs:      docker compose logs -f pipeline
start "" "http://localhost:8010/"
pause
