@echo off
echo ============================================
echo CT600 Vision Inspection System - Docker
echo ============================================
echo.

REM Check if Docker is running
docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Docker is not running!
    echo Please start Docker Desktop and try again.
    echo.
    pause
    exit /b 1
)

echo Docker is running... OK
echo.

REM Check if this is first run
docker images | findstr "ct600-backend" >nul 2>&1
if %errorlevel% neq 0 (
    echo First time setup detected. Building images...
    echo This may take 5-10 minutes...
    echo.
    docker-compose up --build -d
) else (
    echo Starting containers...
    echo.
    docker-compose up -d
)

REM Wait for services to be ready
echo.
echo Waiting for services to start...
timeout /t 5 /nobreak >nul

REM Check status
echo.
echo Checking service status...
docker-compose ps

echo.
echo ============================================
echo Application is running!
echo ============================================
echo.
echo Access the application at:
echo   Vision Inspection: http://localhost:5001/vision-inspection
echo   Frontend:          http://localhost:5001
echo   Backend API:       http://localhost:5000
echo.
echo To view logs:        docker-compose logs -f
echo To stop:            docker-compose down
echo.
pause

