<#
.SYNOPSIS
    VoxScribe 图形界面烟雾测试 + 截图工具。

.DESCRIPTION
    后台启动 `python -m voxscribe`（通过 VOXSCRIBE_GUI_AUTOCLOSE_MS 让界面在指定
    时间后自动退出），等待主窗口出现后把窗口区域截图保存到
    data/screenshots/gui.png，最后等待进程退出并返回其退出码。

.PARAMETER DelayMs
    窗口出现后等待多少毫秒再截图（默认 2000，等待界面布局稳定）。

.PARAMETER AutoCloseMs
    传给 GUI 的自动关闭时间（毫秒，默认 10000）；截图完成后再等进程退出。

.PARAMETER OutFile
    截图输出路径，默认 <项目根>/data/screenshots/gui.png。

.PARAMETER FullScreen
    截取整个屏幕而不是仅截取窗口。

.PARAMETER PythonArgs
    可选：自定义 Python 启动参数（默认 @('-m','voxscribe')），便于截图其他入口/演示脚本。

.PARAMETER TitleMatch
    可选：按窗口标题包含的关键字查找窗口（默认「语声成文」），用于截取设置等对话框。

.EXAMPLE
    pwsh -File tools/screenshot_gui.ps1
    pwsh -File tools/screenshot_gui.ps1 -DelayMs 2500 -OutFile data/screenshots/gui_2.png
#>
[CmdletBinding()]
param(
    [int]$DelayMs = 2000,
    [int]$AutoCloseMs = 10000,
    [string]$OutFile = "",
    [switch]$FullScreen,
    [string]$TitleMatch = "语声成文",
    [string[]]$PythonArgs = @()
)

$ErrorActionPreference = 'Stop'

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path $Python)) {
    throw "找不到虚拟环境 Python：$Python（请先在项目根目录运行 setup.bat）"
}
if ([string]::IsNullOrWhiteSpace($OutFile)) {
    $OutFile = Join-Path $ProjectRoot 'data\screenshots\gui.png'
}
$OutFile = [System.IO.Path]::GetFullPath($OutFile)
$OutDir = Split-Path -Parent $OutFile
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

Add-Type -AssemblyName System.Drawing

# 让本进程 DPI 感知：截图拿到物理像素、GetWindowRect 坐标与之一致。
try {
    $nativeSource = @'
using System;
using System.Text;
using System.Runtime.InteropServices;

namespace VoxScribe
{
    public struct RECT
    {
        public int Left;
        public int Top;
        public int Right;
        public int Bottom;
    }

    public class Win32
    {
        public delegate bool EnumWindowsProc(IntPtr hWnd, IntPtr lParam);

        [DllImport("user32.dll")]
        public static extern bool SetProcessDPIAware();

        [DllImport("user32.dll")]
        public static extern bool SetProcessDpiAwarenessContext(IntPtr value);

        [DllImport("user32.dll")]
        public static extern bool EnumWindows(EnumWindowsProc callback, IntPtr lParam);

        [DllImport("user32.dll", CharSet = CharSet.Unicode)]
        public static extern int GetWindowTextW(IntPtr hWnd, StringBuilder text, int maxCount);

        [DllImport("user32.dll")]
        public static extern bool IsWindowVisible(IntPtr hWnd);

        [DllImport("user32.dll")]
        public static extern bool GetWindowRect(IntPtr hWnd, out RECT rect);

        [DllImport("user32.dll")]
        public static extern int GetSystemMetrics(int index);

        [DllImport("dwmapi.dll")]
        public static extern int DwmGetWindowAttribute(IntPtr hWnd, int attribute, out RECT value, int size);

        public static bool EnableDpiAwareness()
        {
            // -4 = DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2，失败再回退旧 API
            try
            {
                if (SetProcessDpiAwarenessContext(new IntPtr(-4))) return true;
            }
            catch (Exception) { }
            return SetProcessDPIAware();
        }

        public static IntPtr FindWindowByTitle(string keyword)
        {
            IntPtr found = IntPtr.Zero;
            EnumWindows(delegate(IntPtr hWnd, IntPtr lParam)
            {
                if (!IsWindowVisible(hWnd)) return true;
                StringBuilder buffer = new StringBuilder(512);
                GetWindowTextW(hWnd, buffer, buffer.Capacity);
                if (buffer.ToString().Contains(keyword))
                {
                    found = hWnd;
                    return false;
                }
                return true;
            }, IntPtr.Zero);
            return found;
        }

        public static int[] GetRectArray(IntPtr hWnd)
        {
            RECT rect;
            if (!GetWindowRect(hWnd, out rect)) return new int[0];
            return new int[] { rect.Left, rect.Top, rect.Right, rect.Bottom };
        }

        // 可见窗口边界：排除 Win10/11 的窗口阴影与不可见缩放宽边
        public static int[] GetVisibleRectArray(IntPtr hWnd)
        {
            try
            {
                RECT rect;
                int hr = DwmGetWindowAttribute(hWnd, 9, out rect, Marshal.SizeOf(typeof(RECT)));
                if (hr == 0 && rect.Right > rect.Left && rect.Bottom > rect.Top)
                    return new int[] { rect.Left, rect.Top, rect.Right, rect.Bottom };
            }
            catch (Exception) { }
            return GetRectArray(hWnd);
        }

        public static int[] GetScreenSize()
        {
            return new int[] { GetSystemMetrics(0), GetSystemMetrics(1) }; // SM_CXSCREEN / SM_CYSCREEN
        }
    }
}
'@
    Add-Type -TypeDefinition $nativeSource -ErrorAction Stop
    $dpiAware = [VoxScribe.Win32]::EnableDpiAwareness()
    if (-not $dpiAware) {
        Write-Warning "开启 DPI 感知失败，截图可能被系统缩放（仍可用于检查布局）"
    }
} catch {
    Write-Warning "加载 Win32 截图辅助失败：$_"
    throw
}

$PreviousAutoClose = $env:VOXSCRIBE_GUI_AUTOCLOSE_MS
$env:VOXSCRIBE_GUI_AUTOCLOSE_MS = "$AutoCloseMs"
$process = $null
$screenshotOk = $false
$exitCode = 1
try {
    if (-not $PythonArgs -or $PythonArgs.Count -eq 0) {
        $PythonArgs = @('-m', 'voxscribe')
    }
    Write-Host ("启动 GUI：{0} {1}（自动关闭 {2}ms）" -f $Python, ($PythonArgs -join ' '), $AutoCloseMs)
    $process = Start-Process -FilePath $Python `
        -ArgumentList $PythonArgs `
        -WorkingDirectory $ProjectRoot `
        -PassThru -WindowStyle Minimized

    # 等待主窗口出现（标题包含指定关键字，默认「语声成文」），最多 20 秒
    $deadline = (Get-Date).AddSeconds(20)
    $hwnd = [IntPtr]::Zero
    while ((Get-Date) -lt $deadline) {
        Start-Sleep -Milliseconds 250
        $hwnd = [VoxScribe.Win32]::FindWindowByTitle($TitleMatch)
        if ($hwnd -ne [IntPtr]::Zero) { break }
        if ($process.HasExited) { break }
    }
    if ($hwnd -eq [IntPtr]::Zero) {
        throw "20 秒内没有找到窗口（标题包含：$TitleMatch）"
    }
    Write-Host "已找到主窗口，等待 ${DelayMs}ms 后截图…"
    Start-Sleep -Milliseconds $DelayMs

    if ($FullScreen) {
        $bounds = [VoxScribe.Win32]::GetScreenSize()
        $left = 0; $top = 0
        $width = [int]$bounds[0]; $height = [int]$bounds[1]
    } else {
        $rect = [VoxScribe.Win32]::GetVisibleRectArray($hwnd)
        if ($rect.Count -ne 4) { throw "无法获取窗口位置" }
        $left = [int]$rect[0]; $top = [int]$rect[1]
        $width = [int]$rect[2] - $left
        $height = [int]$rect[3] - $top
        $screen = [VoxScribe.Win32]::GetScreenSize()
        # 夹取到屏幕范围内，避免窗口贴边时截图越界
        if ($left -lt 0) { $width += $left; $left = 0 }
        if ($top -lt 0) { $height += $top; $top = 0 }
        $width = [Math]::Min($width, [int]$screen[0] - $left)
        $height = [Math]::Min($height, [int]$screen[1] - $top)
    }
    if ($width -le 0 -or $height -le 0) { throw "窗口尺寸无效：${width}x${height}" }

    $bitmap = New-Object System.Drawing.Bitmap([int]$width, [int]$height)
    $graphics = [System.Drawing.Graphics]::FromImage($bitmap)
    try {
        $graphics.CopyFromScreen([int]$left, [int]$top, 0, 0, $bitmap.Size)
        $bitmap.Save($OutFile, [System.Drawing.Imaging.ImageFormat]::Png)
    } finally {
        $graphics.Dispose()
        $bitmap.Dispose()
    }
    $info = Get-Item $OutFile
    Write-Host ("截图已保存：{0}（{1}x{2}，{3} bytes）" -f $info.FullName, $width, $height, $info.Length)
    $screenshotOk = $true
} finally {
    if ($null -ne $PreviousAutoClose) {
        $env:VOXSCRIBE_GUI_AUTOCLOSE_MS = $PreviousAutoClose
    } else {
        Remove-Item Env:VOXSCRIBE_GUI_AUTOCLOSE_MS -ErrorAction SilentlyContinue
    }
    if ($null -ne $process) {
        $timeoutMs = $AutoCloseMs + 20000
        if (-not $process.WaitForExit($timeoutMs)) {
            Write-Warning "GUI 进程未在预期时间内退出，强制结束。"
            try { $process.Kill($true) } catch { }
            $process.WaitForExit(5000) | Out-Null
        }
        $exitCode = $process.ExitCode
        Write-Host "GUI 进程退出码：$exitCode"
    }
}

if (-not $screenshotOk) { exit 1 }
if ($exitCode -ne 0) {
    Write-Error "GUI 进程退出码异常：$exitCode（截图已保存，可配合 data/logs/app.log 排查）"
    exit 1
}
Write-Host "烟雾测试通过：界面成功启动、截图成功、自动关闭退出码 0。"
exit 0
