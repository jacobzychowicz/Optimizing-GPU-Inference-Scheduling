"""Build the standalone app for this operating system with PyInstaller.

    pip install pyinstaller -r requirements-gui.txt
    python build.py

On Windows: dist/GPU-Inference-Scheduler.exe
On Linux:   dist/GPU-Inference-Scheduler-linux-x86_64 (or -aarch64)

PyInstaller can't cross-compile, so build each one on its own OS (WSL works for
Linux). The release workflow in .github/workflows/release.yml builds both.

Both open the desktop window when started with no arguments and act as the full
command line with arguments (`... hosts test gpu-box`, `... token-mix --live`).
defaults/ and the gpu_bench sources are bundled; the sources are what gets sent to
SSH hosts, so either build can run on a Linux GPU server that has Python 3.8+.
Key logins use the system `ssh` command; password logins use the bundled paramiko.
PyTorch is not bundled, so synthetic-gpu and xtts run live only on SSH hosts or from
a Python install.
"""

import os
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WINDOWS = sys.platform == "win32"
SEP = os.pathsep

# Qt modules the window never uses; leaving them out keeps the app small.
EXCLUDES = [
    "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets", "PySide6.QtWebEngineQuick",
    "PySide6.Qt3DCore", "PySide6.Qt3DRender", "PySide6.QtQuick", "PySide6.QtQml",
    "PySide6.QtQuick3D", "PySide6.QtMultimedia", "PySide6.QtPdf", "PySide6.QtCharts",
    "PySide6.QtDataVisualization", "PySide6.QtBluetooth", "PySide6.QtNetwork",
    "PySide6.QtOpenGL", "PySide6.QtSql", "PySide6.QtTest", "PySide6.QtSvg",
    "tkinter", "unittest", "torch", "numpy",
]


def app_name():
    if WINDOWS:
        return "GPU-Inference-Scheduler"
    machine = platform.machine().lower() or "unknown"
    machine = {"amd64": "x86_64", "arm64": "aarch64"}.get(machine, machine)
    system = "macos" if sys.platform == "darwin" else "linux"
    return f"GPU-Inference-Scheduler-{system}-{machine}"


def main():
    name = app_name()
    work = ROOT / "build" / ("windows" if WINDOWS else sys.platform)
    args = [
        sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", "--onefile",
        "--name", name,
        "--distpath", str(ROOT / "dist"),
        "--workpath", str(work),
        "--specpath", str(work),
        "--add-data", f"{ROOT / 'defaults'}{SEP}defaults",
        "--add-data", f"{ROOT / 'gpu_bench'}{SEP}gpu_bench",
        "--collect-submodules", "gpu_bench",
        # password logins: SSH client and the credential store backends
        "--hidden-import", "paramiko",
        "--collect-submodules", "keyring.backends",
        "--copy-metadata", "keyring",
    ]
    if WINDOWS:
        # A console program, so the command line works from cmd and PowerShell. When it is
        # double-clicked, the console it was given is hidden at once and only the window shows.
        args += ["--hide-console", "hide-early", "--hidden-import", "win32ctypes.core"]
        icon = ROOT / "gpu_bench" / "assets" / "icon.ico"
        if icon.is_file():
            args += ["--icon", str(icon)]
    for module in EXCLUDES:
        args += ["--exclude-module", module]
    args.append(str(ROOT / "launch_app.py"))
    subprocess.check_call(args, cwd=ROOT)
    built = ROOT / "dist" / (name + (".exe" if WINDOWS else ""))
    print(f"built {built} ({built.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
