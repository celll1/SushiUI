@echo off
echo ==========================================
echo   Stable Diffusion WebUI Launcher
echo ==========================================
echo.

:: Check if venv exists
if exist "venv\Scripts\activate.bat" (
    echo [INFO] Virtual environment already exists
) else (
    echo [INFO] Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment
        pause
        exit /b 1
    )
    echo [SUCCESS] Virtual environment created
)

:: Activate venv
echo [INFO] Activating virtual environment...
call venv\Scripts\activate.bat

:: Check if backend dependencies are installed
echo [INFO] Checking backend dependencies...
python -c "import fastapi" 2>nul
if errorlevel 1 (
    echo [INFO] Installing backend dependencies...
    cd backend
    pip install -r requirements.txt
    if errorlevel 1 (
        echo [ERROR] Failed to install dependencies
        cd ..
        pause
        exit /b 1
    )
    cd ..
    echo [SUCCESS] Backend dependencies installed
) else (
    echo [INFO] Backend dependencies already installed
)

:: Check if frontend dependencies are installed
echo [INFO] Checking frontend dependencies...
if exist "frontend\node_modules" (
    echo [INFO] Frontend dependencies already installed
) else (
    echo [INFO] Installing frontend dependencies...
    cd frontend
    call npm install
    if errorlevel 1 (
        echo [ERROR] npm install failed
        cd ..
        pause
        exit /b 1
    )
    cd ..
    echo [SUCCESS] Frontend dependencies installed
)

:: Create necessary directories
echo [INFO] Creating necessary directories...
if not exist "models" mkdir models
if not exist "outputs" mkdir outputs
if not exist "thumbnails" mkdir thumbnails

:: Create .env file if not exists
if not exist "backend\.env" (
    if exist ".env.example" (
        echo [INFO] Creating .env file...
        copy .env.example backend\.env > nul
        echo [SUCCESS] .env file created
    )
)

echo.
echo ==========================================
echo   Starting Servers
echo ==========================================
echo.
echo [INFO] Backend: http://localhost:8000
echo [INFO] Frontend: http://localhost:3000
echo.
echo [NOTE] Press Ctrl+C to stop servers
echo.

:: Start backend in background
echo [INFO] Starting backend server...
start "SD WebUI Backend" cmd /k "cd /d %~dp0 && venv\Scripts\activate.bat && cd backend && python main.py"

:: Wait a bit for backend to start
timeout /t 3 /nobreak > nul

:: Start frontend
echo [INFO] Starting frontend server...
cd frontend
start "SD WebUI Frontend" cmd /k "npm run dev"
cd ..

echo.
echo [SUCCESS] Servers started successfully!
echo.
echo Please open http://localhost:3000 in your browser
echo.
pause
