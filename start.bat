@echo off
setlocal enabledelayedexpansion
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

:: Check if backend dependencies need to be installed/updated
echo [INFO] Checking backend dependencies...
set INSTALL_DEPS=0

:: Check if .requirements_hash exists
if not exist "backend\.requirements_hash" (
    set INSTALL_DEPS=1
) else (
    :: Calculate current requirements.txt hash
    for /f %%A in ('certutil -hashfile backend\requirements.txt SHA256 ^| findstr /v "hash"') do set CURRENT_HASH=%%A

    :: Read stored hash
    set /p STORED_HASH=<backend\.requirements_hash

    :: Compare hashes
    if not "!CURRENT_HASH!"=="!STORED_HASH!" (
        echo [INFO] requirements.txt has been updated
        set INSTALL_DEPS=1
    )
)

if !INSTALL_DEPS!==1 (
    echo [INFO] Installing/updating backend dependencies...
    cd backend
    pip install -r requirements.txt
    if errorlevel 1 (
        echo [ERROR] Failed to install dependencies
        cd ..
        pause
        exit /b 1
    )

    :: Store hash of requirements.txt
    for /f %%A in ('certutil -hashfile requirements.txt SHA256 ^| findstr /v "hash"') do echo %%A > .requirements_hash

    cd ..
    echo [SUCCESS] Backend dependencies installed
) else (
    echo [INFO] Backend dependencies are up to date
)

:: Check if frontend dependencies need to be installed/updated
echo [INFO] Checking frontend dependencies...
set INSTALL_FRONTEND=0

:: Check if node_modules exists
if not exist "frontend\node_modules" (
    set INSTALL_FRONTEND=1
) else (
    :: Check if .package_hash exists
    if not exist "frontend\.package_hash" (
        set INSTALL_FRONTEND=1
    ) else (
        :: Calculate current package.json hash
        for /f %%A in ('certutil -hashfile frontend\package.json SHA256 ^| findstr /v "hash"') do set CURRENT_PKG_HASH=%%A

        :: Read stored hash
        set /p STORED_PKG_HASH=<frontend\.package_hash

        :: Compare hashes
        if not "!CURRENT_PKG_HASH!"=="!STORED_PKG_HASH!" (
            echo [INFO] package.json has been updated
            set INSTALL_FRONTEND=1
        )
    )
)

if !INSTALL_FRONTEND!==1 (
    echo [INFO] Installing/updating frontend dependencies...
    cd frontend
    call npm install
    if errorlevel 1 (
        echo [ERROR] npm install failed
        cd ..
        pause
        exit /b 1
    )

    :: Store hash of package.json
    for /f %%A in ('certutil -hashfile package.json SHA256 ^| findstr /v "hash"') do echo %%A > .package_hash

    cd ..
    echo [SUCCESS] Frontend dependencies installed
) else (
    echo [INFO] Frontend dependencies are up to date
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
