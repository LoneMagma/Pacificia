@echo off
setlocal enabledelayedexpansion
title Pacificia
cd /d "%~dp0"

REM ─────────────────────────────────────────────
REM  Pacificia Launcher
REM  Double-click to run. Handles setup automatically.
REM ─────────────────────────────────────────────

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo  Python not found.
    echo.
    echo  Install Python 3.8+ from https://www.python.org/downloads/
    echo  Make sure to check "Add Python to PATH" during install.
    echo.
    pause
    exit /b 1
)

REM First run: virtual environment doesn't exist yet
if not exist "env\Scripts\activate.bat" (
    echo  First run — setting up Pacificia...
    echo.
    call setup.bat
    exit /b
)

REM Already set up — check API key
if exist ".env" (
    findstr /C:"your_groq_api_key_here" .env >nul 2>&1
    if not errorlevel 1 (
        echo.
        echo  API key not configured.
        echo  Edit .env and add your Groq key, then run this again.
        echo.
        echo  Get a free key at: https://console.groq.com/keys
        echo.
        set /p OPEN="Open .env now? (Y/N): "
        if /i "!OPEN!"=="Y" start notepad .env
        pause
        exit /b 1
    )
) else (
    echo.
    echo  No .env file found. Creating from template...
    if exist ".env.example" (
        copy .env.example .env >nul
    ) else (
        echo GROQ_API_KEY=your_groq_api_key_here > .env
    )
    echo  Edit .env and add your Groq key, then run this again.
    start notepad .env
    pause
    exit /b 1
)

REM Launch
call env\Scripts\activate.bat
python pacificia.py
if errorlevel 1 (
    echo.
    echo  Pacificia exited with an error.
    pause
)
