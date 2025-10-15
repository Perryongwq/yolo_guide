@echo off
echo Starting Backend and Frontend Applications...
echo.
echo Starting Backend on http://localhost:5000...
start cmd /k "cd backend && python app_fastapi.py"

timeout /t 2 /nobreak >nul

echo Starting Frontend on http://localhost:5001...
start cmd /k "cd frontend && python app.py"

echo.
echo Both applications are starting in separate windows.
echo Backend: http://localhost:5000
echo Frontend: http://localhost:5001
echo Vision Inspection: http://localhost:5001/vision-inspection
echo.
pause

