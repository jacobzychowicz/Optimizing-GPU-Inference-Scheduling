"""Build GPU-Inference-Scheduler.exe with PyInstaller.

    pip install pyinstaller PySide6
    python build_exe.py

The .exe opens the desktop window. defaults/ and the gpu_bench sources are bundled
(the sources are needed to send the app to SSH hosts). PyTorch is not bundled, so
the synthetic-gpu model runs live only on SSH hosts or from a Python install.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NAME = "GPU-Inference-Scheduler"
SEP = os.pathsep

# Qt modules the window never uses; leaving them out keeps the .exe small.
EXCLUDES = [
    "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets", "PySide6.QtWebEngineQuick",
    "PySide6.Qt3DCore", "PySide6.Qt3DRender", "PySide6.QtQuick", "PySide6.QtQml",
    "PySide6.QtQuick3D", "PySide6.QtMultimedia", "PySide6.QtPdf", "PySide6.QtCharts",
    "PySide6.QtDataVisualization", "PySide6.QtBluetooth", "PySide6.QtNetwork",
    "PySide6.QtOpenGL", "PySide6.QtSql", "PySide6.QtTest", "PySide6.QtSvg",
    "tkinter", "unittest", "torch", "numpy",
]


def main():
    args = [
        sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", "--onefile", "--windowed",
        "--name", NAME,
        "--distpath", str(ROOT / "dist"),
        "--workpath", str(ROOT / "build"),
        "--specpath", str(ROOT / "build"),
        "--add-data", f"{ROOT / 'defaults'}{SEP}defaults",
        "--add-data", f"{ROOT / 'gpu_bench'}{SEP}gpu_bench",
        "--hidden-import", "gpu_bench.gui",
    ]
    for module in EXCLUDES:
        args += ["--exclude-module", module]
    args.append(str(ROOT / "launch_gui.py"))
    subprocess.check_call(args, cwd=ROOT)
    built = ROOT / "dist" / f"{NAME}.exe"
    shutil.copy2(built, ROOT / f"{NAME}.exe")
    print(f"built {ROOT / (NAME + '.exe')} ({built.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
