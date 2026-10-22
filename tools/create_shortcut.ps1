# 在桌面创建「语声成文 VoxScribe」快捷方式（无黑框启动）
$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$exe = Join-Path $Root "VoxScribe.exe"
if (-not (Test-Path $exe)) {
    # 首次运行：先编译无黑框启动器
    $build = Join-Path $Root "tools\build_launcher.ps1"
    if (Test-Path $build) {
        try {
            & $build | Out-Host
        }
        catch {
            Write-Host "构建启动器失败，将回退到 run.vbs。" -ForegroundColor Yellow
        }
    }
}
if (-not (Test-Path $exe)) {
    Write-Host "未找到启动器：$exe" -ForegroundColor Yellow
    Write-Host "请先运行 setup.bat，或运行 tools\build_launcher.ps1。" -ForegroundColor Yellow
    exit 1
}

$desktop = [Environment]::GetFolderPath("Desktop")
$shortcutPath = Join-Path $desktop "语声成文 VoxScribe.lnk"

$shell = New-Object -ComObject WScript.Shell
$shortcut = $shell.CreateShortcut($shortcutPath)
$shortcut.TargetPath = $exe
$shortcut.Arguments = ""
$shortcut.WorkingDirectory = $Root
$shortcut.Description = "语声成文 VoxScribe：Alt+~ 语音输入，自动整理成文字"
$shortcut.WindowStyle = 1
$shortcut.IconLocation = "$exe,0"
$shortcut.Save()

Write-Host "已创建桌面快捷方式：$shortcutPath" -ForegroundColor Green
Write-Host "以后双击桌面图标即可启动（无黑色命令行窗口）。"
