"""Command line entry. Prints plain text and never prompts, so an SSH session can run it."""

import sys

from gpu_bench.cli import main

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BrokenPipeError:
        try:
            sys.stdout.close()
        except Exception:
            pass
        raise SystemExit(0)
