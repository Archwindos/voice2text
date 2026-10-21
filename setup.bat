@echo off
setlocal
cd /d "%~dp0"
chcp 65001 >nul
where pwsh >nul 2>nul
if %errorlevel%==0 (
    pwsh -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\setup.ps1" %*
) else (
    powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\setup.ps1" %*
)
echo.
pause
