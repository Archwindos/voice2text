' VoxScribe silent launcher for Windows.
' 由桌面快捷方式调用：启动项目根目录的 VoxScribe.exe（Windows GUI 子系统，
' 不会创建任何控制台/黑色命令行窗口）。VoxScribe.exe 缺失时回退到 venv。
Option Explicit

Dim fso, shell, baseDir, exe, command
Set fso = CreateObject("Scripting.FileSystemObject")
Set shell = CreateObject("WScript.Shell")

baseDir = fso.GetParentFolderName(WScript.ScriptFullName)
exe = fso.BuildPath(baseDir, "VoxScribe.exe")
If Not fso.FileExists(exe) Then
    exe = fso.BuildPath(baseDir, ".venv\Scripts\pythonw.exe")
End If

If Not fso.FileExists(exe) Then
    MsgBox "VoxScribe is not installed yet. Please run setup.bat first.", vbExclamation, "VoxScribe"
    WScript.Quit 1
End If

shell.CurrentDirectory = baseDir
command = """" & exe & """"
shell.Run command, 0, False
