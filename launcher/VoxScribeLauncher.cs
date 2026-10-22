using System;
using System.Diagnostics;
using System.IO;
using System.Windows.Forms;

static class VoxScribeLauncher
{
    static string ReadVenvHome(string venvDir)
    {
        string cfg = Path.Combine(venvDir, "pyvenv.cfg");
        if (!File.Exists(cfg))
        {
            return null;
        }
        foreach (string raw in File.ReadAllLines(cfg))
        {
            string line = raw.Trim();
            if (line.StartsWith("home", StringComparison.OrdinalIgnoreCase))
            {
                int idx = line.IndexOf('=');
                if (idx > 0)
                {
                    string value = line.Substring(idx + 1).Trim();
                    if (Directory.Exists(value))
                    {
                        return value;
                    }
                }
            }
        }
        return null;
    }

    [STAThread]
    static int Main()
    {
        string baseDir = AppDomain.CurrentDomain.BaseDirectory.TrimEnd(Path.DirectorySeparatorChar);
        string venvDir = Path.Combine(baseDir, ".venv");
        string pythonw = Path.Combine(venvDir, @"Scripts\pythonw.exe");

        // 使用 venv 背后的真实 pythonw.exe（窗口子系统，不会创建黑色命令行窗口）
        string home = ReadVenvHome(venvDir);
        if (home != null)
        {
            string candidate = Path.Combine(home, "pythonw.exe");
            if (File.Exists(candidate))
            {
                pythonw = candidate;
            }
        }

        if (!File.Exists(pythonw))
        {
            MessageBox.Show(
                "VoxScribe is not installed yet. Please run setup.bat first.",
                "VoxScribe",
                MessageBoxButtons.OK,
                MessageBoxIcon.Warning
            );
            return 1;
        }

        var psi = new ProcessStartInfo(pythonw, "-m voxscribe");
        psi.WorkingDirectory = baseDir;
        psi.UseShellExecute = false;
        psi.CreateNoWindow = true;
        psi.EnvironmentVariables["VIRTUAL_ENV"] = venvDir;
        psi.EnvironmentVariables["PYTHONPATH"] = Path.Combine(venvDir, @"Lib\site-packages");
        psi.EnvironmentVariables["PYTHONUTF8"] = "1";
        psi.EnvironmentVariables["PYTHONIOENCODING"] = "utf-8";

        try
        {
            Process.Start(psi);
            return 0;
        }
        catch (Exception ex)
        {
            MessageBox.Show(
                "Failed to start VoxScribe:\n" + ex.Message,
                "VoxScribe",
                MessageBoxButtons.OK,
                MessageBoxIcon.Error
            );
            return 2;
        }
    }
}
