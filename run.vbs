' VoxScribe silent launcher for Windows.
' Double-click this file to start the app without any black console window.
' The app itself lives in the system tray and can stay in background.
Option Explicit

Dim fso, shell, baseDir, pythonw, command
Set fso = CreateObject("Scripting.FileSystemObject")
Set shell = CreateObject("WScript.Shell")

baseDir = fso.GetParentFolderName(WScript.ScriptFullName)
pythonw = fso.BuildPath(baseDir, ".venv\Scripts\pythonw.exe")

If Not fso.FileExists(pythonw) Then
    MsgBox "VoxScribe is not installed yet. Please run setup.bat first.", vbExclamation, "VoxScribe"
    WScript.Quit 1
End If

shell.CurrentDirectory = baseDir
command = """" & pythonw & """ -m voxscribe"
shell.Run command, 0, False
