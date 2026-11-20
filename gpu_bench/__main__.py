"""Command line entry. Prints plain text and never prompts, so an SSH session can run it."""

import argparse
import sys

from gpu_bench.compare import compare, default_schedulers, default_workloads, load_json, model_for


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Compare GPU inference schedulers and print the results.",
    )
    parser.add_argument(
        "--workload",
        action="append",
        help="Workload JSON file. Repeat for more than one. Default: defaults/workloads.",
    )
    parser.add_argument(
        "--model",
        help="Model JSON file. Default: the model named by each workload.",
    )
    parser.add_argument(
        "--scheduler",
        action="append",
        help="Scheduler Python file. Repeat to compare several. Default: defaults/schedulers.",
    )
    args = parser.parse_args(argv)

    workload_paths = args.workload or [str(path) for path in default_workloads()]
    scheduler_paths = args.scheduler or [str(path) for path in default_schedulers()]
    if not workload_paths:
        print("error: no workload files found", file=sys.stderr)
        return 1
    if not scheduler_paths:
        print("error: no scheduler files found", file=sys.stderr)
        return 1

    from pathlib import Path

    schedulers = [Path(path) for path in scheduler_paths]
    blocks = []
    try:
        for workload_path in workload_paths:
            workload = load_json(workload_path)
            model_path = model_for(workload, args.model)
            blocks.append(compare(workload_path, model_path, schedulers))
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print("\n\n".join(blocks))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BrokenPipeError:
        sys.stdout.close()
        raise SystemExit(0)
