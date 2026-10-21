@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\pythonw.exe" goto missing
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
start "" ".venv\Scripts\pythonw.exe" -m voxscribe
exit /b 0

:missing
echo.
echo [VoxScribe] Dependencies are not installed yet.
echo Please double-click setup.bat first, then run this file again.
echo.
pause
exit /b 1
