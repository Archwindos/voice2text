@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" goto missing
chcp 65001 >nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
".venv\Scripts\python.exe" -m voxscribe --verbose
if errorlevel 1 (
    echo.
    echo The app exited with an error. Log file: data\logs\app.log
    pause
)
exit /b 0

:missing
echo.
echo [VoxScribe] Dependencies are not installed yet.
echo Please double-click setup.bat first.
echo.
pause
exit /b 1
