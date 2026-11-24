"""Entry point for the packaged apps: GPU-Inference-Scheduler.exe (Windows) and
GPU-Inference-Scheduler-linux-x86_64 (Linux).

No arguments: opens the desktop window. On Linux with no display (a server over
SSH, for example) it opens the terminal menu instead.
With arguments: the same command line as `python -m gpu_bench`, for example
`GPU-Inference-Scheduler.exe hosts test gpu-box`.
"""

import os
import sys

from gpu_bench.cli import main as cli_main


def has_display():
    if sys.platform.startswith("linux"):
        return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    return True


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv or not has_display():
        return cli_main(argv)
    from gpu_bench.gui import main as gui_main

    return gui_main()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except BrokenPipeError:
        sys.exit(0)
