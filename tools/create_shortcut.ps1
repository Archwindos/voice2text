# 在桌面创建「语声成文 VoxScribe」快捷方式（无黑框启动）
$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$pythonw = Join-Path $Root ".venv\Scripts\pythonw.exe"
if (-not (Test-Path $pythonw)) {
    Write-Host "未找到运行环境：$pythonw" -ForegroundColor Yellow
    Write-Host "请先运行 setup.bat 安装依赖。" -ForegroundColor Yellow
    exit 1
}

$desktop = [Environment]::GetFolderPath("Desktop")
$shortcutPath = Join-Path $desktop "语声成文 VoxScribe.lnk"
$iconPath = Join-Path $Root "assets\voxscribe.ico"

$shell = New-Object -ComObject WScript.Shell
$shortcut = $shell.CreateShortcut($shortcutPath)
$shortcut.TargetPath = $pythonw
$shortcut.Arguments = "-m voxscribe"
$shortcut.WorkingDirectory = $Root
$shortcut.Description = "语声成文 VoxScribe：Alt+~ 语音输入，自动整理成文字"
$shortcut.WindowStyle = 1
if (Test-Path $iconPath) {
    $shortcut.IconLocation = "$iconPath,0"
}
$shortcut.Save()

Write-Host "已创建桌面快捷方式：$shortcutPath" -ForegroundColor Green
Write-Host "以后双击桌面图标即可启动（无黑色命令行窗口）。"
