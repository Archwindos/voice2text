# VoxScribe one-click setup for Windows
# Creates .venv (Python 3.11), installs dependencies, optionally downloads the ASR model.
# ASCII-only output so it works in Windows PowerShell 5.1 and PowerShell 7.
#
# Optional switches:
#   -NoModelDownload   skip the model download prompt entirely
#   -YesModelDownload  download the model without asking

param(
    [switch]$NoModelDownload,
    [switch]$YesModelDownload
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

function Step([string]$Message) {
    Write-Host ""
    Write-Host "== $Message ==" -ForegroundColor Cyan
}

function Fail([string]$Message) {
    Write-Host ""
    Write-Host "SETUP FAILED: $Message" -ForegroundColor Red
    Write-Host "Please read README.md or ask for help." -ForegroundColor Yellow
    exit 1
}

Write-Host "==============================================="
Write-Host " VoxScribe (Yu Sheng Cheng Wen) setup"
Write-Host " Local speech-to-text + DeepSeek writing polish"
Write-Host "==============================================="
Write-Host "Project: $Root"

$VenvPython = Join-Path $Root ".venv\Scripts\python.exe"

# ---------------------------------------------------------------- 1. venv
if (-not (Test-Path $VenvPython)) {
    Step "Creating virtual environment (Python 3.11)"
    $uv = Get-Command uv -ErrorAction SilentlyContinue
    if ($uv) {
        & uv venv .venv --python 3.11
        if ($LASTEXITCODE -ne 0) { Fail "uv venv failed." }
    }
    else {
        $created = $false
        foreach ($candidate in @("py -3.11", "py -3.12", "py -3")) {
            if ($created) { break }
            Write-Host "Trying: $candidate -m venv .venv"
            $parts = $candidate.Split(" ")
            & $parts[0] $parts[1..($parts.Length - 1)] -m venv .venv 2>$null
            if ($LASTEXITCODE -eq 0) { $created = $true }
        }
        if (-not $created) {
            & python -m venv .venv
            if ($LASTEXITCODE -eq 0) { $created = $true }
        }
        if (-not $created) {
            Fail "No usable Python found. Install Python 3.11/3.12 from python.org, or install uv (winget install astral-sh.uv), then run setup.bat again."
        }
    }
}

if (-not (Test-Path $VenvPython)) { Fail "Virtual environment was not created." }

# ---------------------------------------------------------------- 2. deps
Step "Installing dependencies (first run downloads ~250 MB)"
$uv = Get-Command uv -ErrorAction SilentlyContinue
if ($uv) {
    & uv pip install --python $VenvPython -r requirements.txt
    if ($LASTEXITCODE -ne 0) { Fail "Dependency installation failed (uv)." }
}
else {
    & $VenvPython -m pip --version 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "pip is missing in the virtual environment, bootstrapping it..."
        & $VenvPython -m ensurepip --upgrade
    }
    & $VenvPython -m pip install --upgrade pip
    & $VenvPython -m pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) { Fail "Dependency installation failed (pip)." }
}

# ---------------------------------------------------------------- 3. verify
Step "Environment self-check"
& $VenvPython -m voxscribe --check
if ($LASTEXITCODE -ne 0) {
    Write-Host "Some dependencies are missing. Run setup.bat again or check the messages above." -ForegroundColor Yellow
}

# ---------------------------------------------------------------- 4. model
Step "Speech recognition model (optional download)"
if ($NoModelDownload) {
    Write-Host "Skipped (-NoModelDownload). The model will be downloaded automatically on first transcription."
}
else {
    Write-Host "The 'small' model gives good Chinese accuracy (~484 MB download, ~450 MB RAM while transcribing)."
    if ($YesModelDownload) {
        $answer = "y"
    }
    else {
        $answer = Read-Host "Download it now? [Y/n]"
    }
    if ($answer -eq "" -or $answer -match "^[Yy]") {
        & $VenvPython tools\prefetch_model.py --model small
        if ($LASTEXITCODE -ne 0) {
            Write-Host "Direct download failed. Retrying through the Hugging Face mirror (hf-mirror.com)..." -ForegroundColor Yellow
            $env:HF_ENDPOINT = "https://hf-mirror.com"
            $env:HF_HUB_DISABLE_XET = "1"
            & $VenvPython tools\prefetch_model.py --model small
            if ($LASTEXITCODE -ne 0) {
                Write-Host "Model download did not finish. It will be retried automatically on first transcription." -ForegroundColor Yellow
                Write-Host "You can also retry manually later with the mirror:" -ForegroundColor Yellow
                Write-Host '  $env:HF_ENDPOINT="https://hf-mirror.com"; .venv\Scripts\python.exe tools\prefetch_model.py --model small'
            }
        }
    }
    else {
        Write-Host "Skipped. The model will be downloaded automatically on first transcription."
    }
}

Write-Host ""
Write-Host "==============================================="
Write-Host " Setup complete!" -ForegroundColor Green
Write-Host " Double-click run.bat to start the app."
Write-Host " First: click Settings, fill in your DeepSeek API Key."
Write-Host " Get a key at https://platform.deepseek.com/"
Write-Host "==============================================="
Write-Host ""
