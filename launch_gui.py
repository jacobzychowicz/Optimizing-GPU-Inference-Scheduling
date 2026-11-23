"""Entry point for the packaged Windows app (GPU-Inference-Scheduler.exe). Opens the desktop window."""

import sys

from gpu_bench.gui import main

if __name__ == "__main__":
    sys.exit(main())
