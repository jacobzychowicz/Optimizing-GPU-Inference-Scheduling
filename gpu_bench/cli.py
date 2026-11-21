"""Command line for comparing schedulers by name."""

import argparse
import sys
from pathlib import Path

from gpu_bench.catalog import format_catalog, resolve, scheduler_items, workload_items
from gpu_bench.compare import compare, load_json, model_for


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    try:
        return dispatch(argv)
    except (ValueError, RuntimeError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def dispatch(argv):
    if not argv:
        return cmd_run([])

    head, rest = argv[0], argv[1:]
    if head in ("-h", "--help"):
        print(top_help())
        return 0
    if head == "help":
        if not rest:
            print(top_help())
            return 0
        return dispatch([rest[0], "-h", *rest[1:]])
    if head == "list":
        return cmd_list(rest)
    if head == "run":
        return cmd_run(rest)
    return cmd_run(argv)


def cmd_list(argv):
    parser = argparse.ArgumentParser(
        prog=f"{program_name()} list",
        description="Show the workloads, models, and schedulers you can run by name.",
    )
    parser.add_argument(
        "kind",
        nargs="?",
        choices=("workloads", "models", "schedulers"),
        help="show one group instead of all three",
    )
    args = parser.parse_args(argv)
    print(format_catalog(args.kind))
    if args.kind is None:
        print(f"\nRun a comparison with: {program_name()} <workload> [scheduler ...]")
    return 0


def cmd_run(argv):
    prog = program_name()
    parser = argparse.ArgumentParser(
        prog=f"{prog} run",
        description=(
            "Compare schedulers on a workload. "
            f"Each name is a workload, model, or scheduler from `{prog} list`, "
            "or a path to your own file."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            f"  {prog} mixed-lengths\n"
            f"  {prog} mixed-lengths fifo priority\n"
            f"  {prog} token-mix --model example-llm\n"
            f"  {prog} run priority-mix -s fifo,aging\n"
            "\n"
            "Schedulers you name are the only ones that run.\n"
            "Workloads you name are the only ones that run.\n"
            "Leave either out to use every sample of that kind.\n"
            "One --model replaces the model on every workload in the run."
        ),
    )
    parser.add_argument(
        "names",
        nargs="*",
        metavar="name",
        help="workload, model, and scheduler names, or paths to your own files",
    )
    parser.add_argument(
        "-w",
        "--workload",
        action="append",
        default=[],
        help="workload name or JSON file. Repeat to run more than one.",
    )
    parser.add_argument(
        "-s",
        "--scheduler",
        action="append",
        default=[],
        help="scheduler name or Python file. Repeat, or separate names with commas.",
    )
    parser.add_argument(
        "-m",
        "--model",
        help="model name or JSON file. Default: the model named by the workload.",
    )
    args = parser.parse_args(argv)

    workloads = []
    schedulers = []
    positional_models = []
    for token in _flatten(args.names):
        item = resolve(token)
        if item["kind"] == "workload":
            workloads.append(item)
        elif item["kind"] == "scheduler":
            schedulers.append(item)
        else:
            positional_models.append(item)
    for token in _flatten(args.workload):
        workloads.append(resolve(token, "workload"))
    for token in _flatten(args.scheduler):
        schedulers.append(resolve(token, "scheduler"))
    flag_model = resolve(args.model, "model") if args.model else None

    workloads = _dedupe(workloads)
    schedulers = _dedupe(schedulers)
    models = _dedupe(positional_models + ([flag_model] if flag_model else []))
    if len(models) > 1:
        names = " and ".join(item["name"] for item in models)
        raise ValueError(f"pass one model. Got {names}.")
    named_only_a_model = (
        len(positional_models) == 1
        and flag_model is None
        and not workloads
        and not schedulers
        and not args.workload
        and not args.scheduler
    )
    if named_only_a_model:
        model_name = positional_models[0]["name"]
        sample = next((item["name"] for item in workload_items() if item["model"] == model_name), None)
        if sample:
            example = f"{prog} {sample} {model_name}"
        else:
            example = f"{prog} WORKLOAD --model {model_name}"
        raise ValueError(f"{model_name} is a model. Name a workload too, for example: {example}")

    if not workloads:
        workloads = workload_items()
    if not schedulers:
        schedulers = list(scheduler_items())
    if not workloads:
        raise ValueError("no workloads found")
    if not schedulers:
        raise ValueError("no schedulers found")

    model_item = models[0] if models else None
    blocks = []
    chosen = [(item["name"], item["path"]) for item in schedulers]
    for workload_item in workloads:
        workload_path = workload_item["path"]
        if model_item is not None:
            model_path = model_item["path"]
        else:
            model_path = model_for(load_json(workload_path), None)
        blocks.append(compare(workload_path, model_path, chosen))
    print("\n\n".join(blocks))
    return 0


def program_name():
    invoked = Path(sys.argv[0]).name if sys.argv else ""
    if invoked == "gpu-bench":
        return "gpu-bench"
    return "python -m gpu_bench"


def top_help():
    prog = program_name()
    return f"""usage: {prog} [list|run] [name ...]

Compare schedulers on a workload and print the results.
Names come from `{prog} list`. A path to your own file works anywhere a name works.

  {prog} list
  {prog} mixed-lengths fifo priority
  {prog} token-mix --model example-llm
  {prog} run priority-mix -s fifo,aging

With no names, every sample workload is compared on every sample scheduler.
`{prog} run -h` explains each flag.
"""


def _flatten(values):
    names = []
    for value in values:
        if any(sep in value for sep in ("/", "\\")) or Path(value).exists():
            names.append(value)
            continue
        for part in value.split(","):
            part = part.strip()
            if part:
                names.append(part)
    return names


def _dedupe(items):
    chosen = []
    seen = set()
    for item in items:
        key = str(Path(item["path"]).resolve())
        if key in seen:
            continue
        seen.add(key)
        chosen.append(item)
    return chosen

