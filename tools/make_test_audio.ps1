<#
.SYNOPSIS
    用 Windows 自带的 System.Speech 合成一段中文测试音频。

.DESCRIPTION
    优先使用中文语音 Microsoft Huihui Desktop（zh-CN），
    输出 16kHz / 单声道 / PCM16 的 wav，供 tests/test_asr.py 的
    端到端语音识别测试使用（默认 tests/fixtures/tts_zh.wav）。

    没有中文语音包时会给出明确提示并以非 0 退出码结束；
    测试代码检测到音频缺失时会优雅跳过端到端用例。

.PARAMETER OutFile
    输出文件路径，默认 <项目根>/tests/fixtures/tts_zh.wav。

.PARAMETER Text
    要合成的文本。

.EXAMPLE
    powershell -NoProfile -ExecutionPolicy Bypass -File tools/make_test_audio.ps1

.EXAMPLE
    powershell -NoProfile -ExecutionPolicy Bypass -File tools/make_test_audio.ps1 -OutFile D:\tmp\test.wav
#>
[CmdletBinding()]
param(
    [string]$OutFile = "",
    [string]$Text = "嗯，那个，我们今天主要讨论一下项目的进度问题。前端已经完成了百分之七十，后端还需要两周时间，就是这样。"
)

$ErrorActionPreference = 'Stop'

# ---- 计算输出路径（默认 tests/fixtures/tts_zh.wav）--------------------
$projectRoot = Split-Path -Parent $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($OutFile)) {
    $OutFile = Join-Path $projectRoot 'tests\fixtures\tts_zh.wav'
}
if (-not [System.IO.Path]::IsPathRooted($OutFile)) {
    $OutFile = Join-Path (Get-Location).Path $OutFile
}
$OutFile = [System.IO.Path]::GetFullPath($OutFile)
$outDir = Split-Path -Parent $OutFile
if (-not (Test-Path -LiteralPath $outDir)) {
    New-Item -ItemType Directory -Path $outDir -Force | Out-Null
}

try {
    Add-Type -AssemblyName System.Speech

    $synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
    try {
        # 优先 Huihui（zh-CN），其次任意中文语音
        $voice = $null
        $installed = @($synth.GetInstalledVoices() | Where-Object { $_.Enabled })
        $voice = $installed |
            Where-Object { $_.VoiceInfo.Culture.Name -eq 'zh-CN' -and $_.VoiceInfo.Name -like '*Huihui*' } |
            Select-Object -First 1
        if (-not $voice) {
            $voice = $installed |
                Where-Object { $_.VoiceInfo.Culture.Name -eq 'zh-CN' } |
                Select-Object -First 1
        }
        if (-not $voice) {
            throw ('未找到中文语音。请在 Windows「设置 → 时间和语言 → 语音」中安装中文语音包' +
                '（例如 Microsoft Huihui Desktop），或使用 -Text 之外的英文语音方案。')
        }
        $synth.SelectVoice($voice.VoiceInfo.Name)

        # 16kHz / 单声道 / 16bit（Whisper 原生采样率，避免重采样）
        $format = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(
            16000,
            [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen,
            [System.Speech.AudioFormat.AudioChannel]::Mono
        )
        $synth.SetOutputToWaveFile($OutFile, $format)

        Write-Host "语音：$($voice.VoiceInfo.Name)（$($voice.VoiceInfo.Culture.Name)）"
        Write-Host "文本：$Text"
        Write-Host "正在合成：$OutFile"
        $synth.Speak($Text)
    }
    finally {
        $synth.Dispose()
    }

    if (-not (Test-Path -LiteralPath $OutFile)) {
        throw "合成失败：没有生成文件 $OutFile"
    }
    $size = (Get-Item -LiteralPath $OutFile).Length
    if ($size -le 44) {
        throw "合成失败：输出文件为空（$size 字节）"
    }
    Write-Host ("完成：{0}（{1:N1} KB）" -f $OutFile, ($size / 1KB)) -ForegroundColor Green
    exit 0
}
catch {
    [Console]::Error.WriteLine("生成测试音频失败：$($_.Exception.Message)")
    exit 1
}
