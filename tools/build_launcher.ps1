# Build the tiny silent launcher (VoxScribe.exe, Windows GUI subsystem = no console).
# Requires the .NET Framework C# compiler shipped with Windows.
$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$csc = Join-Path $env:WINDIR "Microsoft.NET\Framework64\v4.0.30319\csc.exe"
if (-not (Test-Path $csc)) {
    $csc = Join-Path $env:WINDIR "Microsoft.NET\Framework\v4.0.30319\csc.exe"
}
if (-not (Test-Path $csc)) {
    Write-Host "C# compiler not found; skip building VoxScribe.exe" -ForegroundColor Yellow
    exit 1
}

$source = Join-Path $Root "launcher\VoxScribeLauncher.cs"
$exe = Join-Path $Root "VoxScribe.exe"
$icon = Join-Path $Root "assets\voxscribe.ico"
$args = @(
    "/nologo",
    "/target:winexe",
    "/optimize+",
    "/out:$exe",
    "/r:System.Windows.Forms.dll",
    $source
)
if (Test-Path $icon) {
    $args = @(
        "/nologo",
        "/target:winexe",
        "/optimize+",
        "/out:$exe",
        "/win32icon:$icon",
        "/r:System.Windows.Forms.dll",
        $source
    )
}

& $csc @args
if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to build VoxScribe.exe" -ForegroundColor Red
    exit 1
}
Write-Host "Built: $exe" -ForegroundColor Green
exit 0
