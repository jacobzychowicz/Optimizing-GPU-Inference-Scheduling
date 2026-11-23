"""Command line for comparing schedulers by name."""

import argparse
import json
import sys
from pathlib import Path

from gpu_bench.catalog import format_catalog, resolve, scheduler_items, workload_items
from gpu_bench.compare import load_json, model_for


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    try:
        return dispatch(argv)
    except KeyboardInterrupt:
        print("stopped", file=sys.stderr)
        return 130
    except (ValueError, RuntimeError, KeyError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def dispatch(argv):
    if not argv:
        if _terminal_can_prompt():
            return _open_menu([])
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
    if head == "menu":
        return _open_menu(rest)
    if head == "list":
        return cmd_list(rest)
    if head == "run":
        return cmd_run(rest)
    if head == "gui":
        return _open_gui(rest)
    if head == "gpus":
        return cmd_gpus(rest)
    if head == "hosts":
        return cmd_hosts(rest)
    if head == "history":
        return cmd_history(rest)
    if head == "generate":
        return cmd_generate(rest)
    if head == "calibrate":
        return cmd_calibrate(rest)
    return cmd_run(argv)


def _terminal_can_prompt():
    return _stdin_is_console() and sys.stdout is not None and sys.stdout.isatty()


def _stdin_is_console():
    """True only when someone can type an answer. On Windows, NUL claims to be a tty, so ask the console."""
    if sys.stdin is None or not sys.stdin.isatty():
        return False
    if sys.platform != "win32":
        return True
    try:
        import ctypes
        import msvcrt

        handle = msvcrt.get_osfhandle(sys.stdin.fileno())
        mode = ctypes.c_uint()
        return bool(ctypes.windll.kernel32.GetConsoleMode(handle, ctypes.byref(mode)))
    except (OSError, ValueError, AttributeError):
        return False


def _open_menu(rest):
    prog = program_name()
    if rest and rest[0] in ("-h", "--help"):
        print(
            f"usage: {prog} menu\n"
            "\n"
            "Open the numbered menu. Pick a model, a workload, and schedulers, then run.\n"
            "Options 6 and 7 choose simulate or live, and this machine or an SSH host."
        )
        return 0
    if rest:
        print("error: menu takes no arguments", file=sys.stderr)
        return 1
    if not _terminal_can_prompt():
        print(
            "error: the menu needs an interactive terminal.\n"
            f"Type a command instead, for example: {prog} mixed-lengths fifo priority",
            file=sys.stderr,
        )
        return 1
    from gpu_bench.menu import run_menu

    return run_menu()


def _open_gui(rest):
    prog = program_name()
    if rest and rest[0] in ("-h", "--help"):
        print(
            f"usage: {prog} gui\n"
            "\n"
            "Open the desktop window. Install PySide6 first: pip install PySide6"
        )
        return 0
    if rest:
        print("error: gui takes no arguments", file=sys.stderr)
        return 1
    from gpu_bench.gui import main as gui_main

    return gui_main()


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
            f"  {prog} token-mix --model synthetic-gpu --live --repeat 3\n"
            f"  {prog} token-mix --live --host gpu-box --save\n"
            "\n"
            "Schedulers you name are the only ones that run.\n"
            "Workloads you name are the only ones that run.\n"
            "Leave either out to use every sample of that kind.\n"
            "One --model replaces the model on every workload in the run.\n"
            "\n"
            "simulate (the default) computes the schedule from the model's runtime formula.\n"
            "live replays the workload on a real clock and runs each request on the\n"
            "model's backend, measuring real latency and GPU use."
        ),
    )
    parser.add_argument(
        "names",
        nargs="*",
        metavar="name",
        help="workload, model, and scheduler names, or paths to your own files",
    )
    parser.add_argument("-w", "--workload", action="append", default=[],
                        help="workload name or JSON file. Repeat to run more than one.")
    parser.add_argument("-s", "--scheduler", action="append", default=[],
                        help="scheduler name or Python file. Repeat, or separate names with commas.")
    parser.add_argument("-m", "--model", help="model name or JSON file. Default: the model named by the workload.")
    parser.add_argument("--mode", choices=("simulate", "live"), default="simulate",
                        help="simulate (default) or live on a real GPU")
    parser.add_argument("--live", action="store_const", const="live", dest="mode", help="same as --mode live")
    parser.add_argument("--host", help="run on a saved SSH host (see `hosts`) or user@host[:port]")
    parser.add_argument("--repeat", type=int, default=1, metavar="N",
                        help="live runs: run each scheduler N times and average (default 1)")
    parser.add_argument("--save", action="store_true", help="keep the result in the run history")
    parser.add_argument("--out", metavar="FILE", help="also write the result to FILE (.json, or .csv for a summary)")
    parser.add_argument("--json", action="store_true", help="print the full result as JSON instead of a table")
    parser.add_argument("--events", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.repeat < 1:
        raise ValueError("--repeat must be 1 or more")

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

    host = None
    if args.host:
        from gpu_bench.store import find_host

        host = prepare_host(find_host(args.host))

    from gpu_bench.runner import format_result

    model_item = models[0] if models else None
    chosen = [(item["name"], item["path"]) for item in schedulers]
    emit = _event_printer() if args.events else _progress_printer(args.mode, host)
    results = []
    for workload_item in workloads:
        workload_path = workload_item["path"]
        if model_item is not None:
            model_path = model_item["path"]
        else:
            model_path = model_for(load_json(workload_path), None)
        results.append(
            with_trust(lambda: run_anywhere(host, workload_path, model_path, chosen, args.mode, args.repeat, emit))
        )

    if args.events:
        return 0
    for result in results:
        if args.save:
            from gpu_bench.store import save_run

            path = save_run(result)
            print(f"saved to {path}", file=sys.stderr)
    if args.out:
        from gpu_bench.store import export

        payload = results[0] if len(results) == 1 else {"runs": results}
        if Path(args.out).suffix.lower() == ".csv" and len(results) > 1:
            raise ValueError("--out .csv takes one workload. Use .json for several.")
        if Path(args.out).suffix.lower() == ".json":
            Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        else:
            export(results[0], args.out)
        print(f"wrote {args.out}", file=sys.stderr)
    if args.json:
        print(json.dumps(results[0] if len(results) == 1 else results, indent=2))
    else:
        print("\n\n".join(format_result(result) for result in results))
    return 0


def run_anywhere(host, workload_path, model_path, schedulers, mode, repeats, emit=None, cancel=None):
    """Run on this machine, or on host over SSH when host is given."""
    if host is None:
        from gpu_bench.runner import execute

        return execute(workload_path, model_path, schedulers, mode, repeats, emit, cancel)
    from gpu_bench.remote import run

    return run(host, workload_path, model_path, schedulers, mode, repeats, emit, cancel)


def prepare_host(host):
    """Attach the password for a password-login host, asking in the terminal if it isn't saved."""
    from gpu_bench.remote import with_password

    return with_password(host, _ask_password)


def _ask_password(host):
    import getpass
    import os

    from gpu_bench.store import describe_host

    env = os.environ.get("GPU_BENCH_SSH_PASSWORD")
    if env:
        return env, False
    if not _stdin_is_console():
        raise ValueError(
            f"{describe_host(host)} logs in with a password. Run this in a terminal to type it, "
            "save it (hosts add NAME user@host --password --save-password), or set GPU_BENCH_SSH_PASSWORD."
        )
    password = getpass.getpass(f"Password for {describe_host(host)}: ")
    if not password:
        return None
    return password, bool(host.get("save_password"))


def with_trust(fn):
    """Call fn. If a password host's server key is new, show its fingerprint and ask before trusting it."""
    from gpu_bench.remote import UnknownHostKey, trust_host_key

    while True:
        try:
            return fn()
        except UnknownHostKey as exc:
            if not _stdin_is_console():
                raise RuntimeError(f"{exc}\nRun `hosts test NAME` in a terminal to confirm it.") from exc
            print(str(exc), file=sys.stderr)
            answer = input("Trust this server and continue? [y/N] ").strip().lower()
            if answer not in ("y", "yes"):
                raise RuntimeError("server not trusted; nothing was sent") from exc
            trust_host_key(exc)


def _event_printer():
    def emit(event):
        sys.stdout.write(json.dumps(event) + "\n")
        sys.stdout.flush()

    return emit


def _progress_printer(mode, host):
    """Live and remote runs take real time, so say what is happening on stderr."""
    if mode == "simulate" and host is None:
        return None

    def emit(event):
        kind = event.get("event")
        if kind == "log":
            print(f"  {event['message']}", file=sys.stderr)
        elif kind == "scheduler_start":
            suffix = f" (run {event['repeat']} of {event['repeats']})" if event.get("repeats", 1) > 1 else ""
            print(f"  {event['scheduler']}{suffix}", file=sys.stderr)
        elif kind == "job":
            job = event["job"]
            print(
                f"    {event['done']}/{event['total']}  {job['id']}  "
                f"waited {job['wait_ms']:.0f} ms, ran {job['service_ms']:.0f} ms",
                file=sys.stderr,
            )

    return emit


def cmd_generate(argv):
    prog = program_name()
    parser = argparse.ArgumentParser(
        prog=f"{prog} generate",
        description="Make a workload file. Saved to ~/.gpu_bench/workloads so it shows up by name.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "presets:\n"
            "  narration  long scripts split into one request per sentence (all arriving at once),\n"
            "             plus single-line retakes from other users who are waiting on them\n"
            "  traffic    requests arriving steadily, in bursts, or ramping up\n"
            "\n"
            "examples:\n"
            f"  {prog} generate narration-busy --users 4 --segments 100 400 --retakes 20\n"
            f"  {prog} generate chat-200 --preset traffic --count 200 --rate 4 --pattern bursty"
        ),
    )
    parser.add_argument("name", help="workload name (also the file name)")
    parser.add_argument("--preset", choices=("narration", "traffic"), default="narration")
    parser.add_argument("-m", "--model", help="model whose runtime fills predicted_duration_ms (and is the default model)")
    parser.add_argument("--seed", type=int, default=7, help="same seed, same workload (default 7)")
    parser.add_argument("--out", help="write here instead of ~/.gpu_bench/workloads/NAME.json")
    parser.add_argument("--force", action="store_true", help="overwrite an existing file")
    narration = parser.add_argument_group("narration")
    narration.add_argument("--users", type=int, default=4, help="people sharing the GPU (default 4)")
    narration.add_argument("--script-users", type=int, help="how many submit a full script (default: half)")
    narration.add_argument("--segments", type=int, nargs=2, default=(100, 400), metavar=("MIN", "MAX"),
                           help="sentences per script (default 100 400)")
    narration.add_argument("--retakes", type=int, default=20, help="single-line retakes (default 20)")
    traffic = parser.add_argument_group("traffic")
    traffic.add_argument("--count", type=int, default=200)
    traffic.add_argument("--rate", type=float, default=4.0, help="requests per second (default 4)")
    traffic.add_argument("--pattern", choices=("steady", "bursty", "ramp"), default="steady")
    traffic.add_argument("--field", choices=("max_tokens", "text"), default="max_tokens",
                         help="request size field: max_tokens (LLM) or text (speech)")
    args = parser.parse_args(argv)

    from gpu_bench import workloads
    from gpu_bench.store import SAFE_NAME, user_dir

    if not SAFE_NAME.match(args.name):
        raise ValueError("a workload name may use letters, digits, dot, dash, and underscore")
    model = model_name = None
    if args.model:
        item = resolve(args.model, "model")
        model, model_name = load_json(item["path"]), item["name"]
    if args.preset == "narration":
        data = workloads.generate_narration(
            args.name, args.users, args.script_users, args.segments, args.retakes,
            model=model, model_name=model_name, seed=args.seed,
        )
    else:
        data = workloads.generate_traffic(
            args.name, args.count, args.rate, args.pattern, args.field,
            model=model, model_name=model_name, seed=args.seed,
        )
    path = Path(args.out) if args.out else user_dir("workloads") / f"{args.name}.json"
    if path.exists() and not args.force:
        raise ValueError(f"{path} already exists. Pass --force to replace it.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    kinds = {}
    for job in data["jobs"]:
        kinds[job.get("kind", "request")] = kinds.get(job.get("kind", "request"), 0) + 1
    detail = ", ".join(f"{count} {kind}" for kind, count in kinds.items())
    print(f"wrote {path}: {len(data['jobs'])} jobs ({detail}), model {data['model']}")
    print(f"run it with: {prog} {args.name} fifo fair_share retake_first")
    return 0


def cmd_calibrate(argv):
    prog = program_name()
    parser = argparse.ArgumentParser(
        prog=f"{prog} calibrate",
        description=(
            "Fit the model's runtime formula to the GPU times a live run measured, then check how "
            "closely a simulation with the fitted model matches the live run."
        ),
    )
    parser.add_argument("run", help="run number from `history`, or a result .json file")
    parser.add_argument("--save", metavar="NAME", nargs="?", const="",
                        help="save the calibrated model to ~/.gpu_bench/models (default name: MODEL-calibrated)")
    args = parser.parse_args(argv)

    from gpu_bench import calibrate, store

    if args.run.isdigit():
        runs = store.list_runs()
        number = int(args.run)
        if not 1 <= number <= len(runs):
            raise ValueError(f"pass a run number from 1 to {len(runs)} (see `{prog} history`)")
        result = store.load_run(runs[number - 1]["path"])
    else:
        result = load_json(args.run)
    coefficients = calibrate.fit(result)
    model = calibrate.calibrated_model(result, coefficients, args.save or None)
    print(calibrate.format_report(coefficients, calibrate.check(result, model)))
    if args.save is not None:
        if not store.SAFE_NAME.match(model["name"]):
            raise ValueError("a model name may use letters, digits, dot, dash, and underscore")
        path = store.user_dir("models") / f"{model['name']}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(model, indent=2), encoding="utf-8")
        print()
        print(f"saved {path}. Simulate with: {prog} WORKLOAD --model {model['name']}")
    return 0


def cmd_gpus(argv):
    parser = argparse.ArgumentParser(
        prog=f"{program_name()} gpus",
        description="Show the GPUs, Python, and PyTorch on this machine or an SSH host.",
    )
    parser.add_argument("--host", help="saved host name or user@host[:port]")
    parser.add_argument("--json", action="store_true", help="print JSON")
    parser.add_argument("--no-torch", action="store_true", help="skip the PyTorch check (faster)")
    args = parser.parse_args(argv)
    if args.host:
        from gpu_bench.remote import probe
        from gpu_bench.store import find_host

        host = prepare_host(find_host(args.host))
        info = with_trust(lambda: probe(host, check_torch=not args.no_torch))
    else:
        from gpu_bench.gpus import probe

        info = probe(check_torch=not args.no_torch)
    if args.json:
        print(json.dumps(info))
    else:
        from gpu_bench.gpus import format_probe

        print(format_probe(info))
    return 0


def cmd_hosts(argv):
    prog = program_name()
    parser = argparse.ArgumentParser(
        prog=f"{prog} hosts",
        description="Save SSH hosts so runs can use --host NAME. Login uses your normal ssh keys.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            f"  {prog} hosts add gpu-box alice@10.0.0.5 --python /opt/venv/bin/python\n"
            f"  {prog} hosts add lab bob@lab.example.edu --password --save-password\n"
            f"  {prog} hosts test gpu-box\n"
            f"  {prog} hosts remove gpu-box"
        ),
    )
    sub = parser.add_subparsers(dest="action")
    sub.add_parser("list", help="show saved hosts")
    add = sub.add_parser("add", help="save a host (or update one with the same name)")
    add.add_argument("name")
    add.add_argument("target", help="user@host, host, or a Host alias from ~/.ssh/config")
    add.add_argument("--port", type=int)
    add.add_argument("--key", help="private key file")
    add.add_argument("--python", default="python3", help="Python on the server (default python3)")
    add.add_argument("--workdir", default=".gpu-bench", help="scratch folder on the server (default ~/.gpu-bench)")
    add.add_argument("-o", "--ssh-option", action="append", default=[], help="extra ssh -o Key=Value")
    add.add_argument("--password", action="store_true",
                     help="log in with a password instead of keys (asked for when connecting)")
    add.add_argument("--save-password", action="store_true",
                     help="with --password: ask for it now and keep it in the system credential store")
    remove = sub.add_parser("remove", help="forget a saved host")
    remove.add_argument("name")
    test = sub.add_parser("test", help="connect and show the server's GPUs")
    test.add_argument("name")
    args = parser.parse_args(argv)

    from gpu_bench import store

    if args.action in (None, "list"):
        hosts = store.load_hosts()
        if not hosts:
            print(f"No saved hosts. Add one with: {prog} hosts add NAME user@host")
            return 0
        for host in hosts:
            if host["auth"] == "password":
                login = "password (saved)" if store.get_saved_password(host["name"]) else "password (asked each time)"
            else:
                login = "ssh keys"
            print(f"  {host['name']:<16} {store.describe_host(host)}  login={login}  python={host['python']}")
        return 0
    if args.action == "add":
        if args.save_password and not args.password:
            raise ValueError("--save-password goes with --password")
        if args.password and (args.key or args.ssh_option):
            raise ValueError("--key and --ssh-option apply to key logins, not --password")
        password = None
        if args.save_password:
            import getpass

            if not store.can_save_passwords():
                raise ValueError("this computer has no credential store to save passwords in. Leave out --save-password.")
            password = getpass.getpass(f"Password for {args.target}: ")
            if not password:
                raise ValueError("no password typed")
        user, _, address = args.target.rpartition("@")
        host = store.save_host(
            {
                "name": args.name,
                "host": address,
                "user": user,
                "port": args.port,
                "key": args.key,
                "python": args.python,
                "workdir": args.workdir,
                "ssh_options": args.ssh_option,
                "auth": "password" if args.password else "key",
                "save_password": args.save_password,
            }
        )
        if password is not None:
            store.set_saved_password(host["name"], password)
        print(f"saved {host['name']} ({store.describe_host(host)}). Check it with: {prog} hosts test {host['name']}")
        return 0
    if args.action == "remove":
        store.remove_host(args.name)
        print(f"removed {args.name}")
        return 0
    if args.action == "test":
        from gpu_bench.gpus import format_probe
        from gpu_bench.remote import probe

        host = prepare_host(store.find_host(args.name))
        print(f"connecting to {store.describe_host(host)} ...", file=sys.stderr)
        print(format_probe(with_trust(lambda: probe(host))))
        return 0
    return 1


def cmd_history(argv):
    parser = argparse.ArgumentParser(
        prog=f"{program_name()} history",
        description="Saved runs (from --save or the desktop window), newest first.",
    )
    parser.add_argument("action", nargs="?", choices=("list", "show", "export"), default="list")
    parser.add_argument("number", nargs="?", type=int, help="run number from the list")
    parser.add_argument("file", nargs="?", help="export: .json, .csv (summary), or .jobs.csv (every job)")
    args = parser.parse_args(argv)

    from gpu_bench import store
    from gpu_bench.runner import format_result

    runs = store.list_runs()
    if args.action == "list":
        if not runs:
            print("No saved runs yet. Add --save to a run, or run from the desktop window.")
            return 0
        for index, row in enumerate(runs, start=1):
            print(
                f"{index:>3}  {row['created']:<19}  {row['workload']:<16} {row['model']:<16} "
                f"{row['mode']:<8} {row['target']:<12} best: {row['best']}"
            )
        return 0
    if args.number is None or not 1 <= args.number <= len(runs):
        raise ValueError(f"pass a run number from 1 to {len(runs)}")
    result = store.load_run(runs[args.number - 1]["path"])
    if args.action == "show":
        print(format_result(result))
        return 0
    if not args.file:
        raise ValueError("pass a file to export to")
    kind = "jobs" if args.file.lower().endswith(".jobs.csv") else "summary"
    store.export(result, args.file, kind)
    print(f"wrote {args.file}")
    return 0


def program_name():
    invoked = Path(sys.argv[0]).name if sys.argv else ""
    if invoked == "gpu-bench":
        return "gpu-bench"
    return "python -m gpu_bench"


def top_help():
    prog = program_name()
    return f"""usage: {prog} [menu|list|run|gpus|hosts|history|generate|calibrate|gui] [name ...]

In a terminal, `{prog}` opens a menu. Choose a model, a workload, and
schedulers, then run. Names and flags still run directly, with no menu.

  {prog}
  {prog} mixed-lengths fifo priority
  {prog} list
  {prog} token-mix --model example-llm
  {prog} token-mix --model synthetic-gpu --live      real GPU, this machine
  {prog} token-mix --live --host gpu-box             real GPU, over SSH
  {prog} gpus [--host gpu-box]                       what GPUs are there
  {prog} hosts add gpu-box alice@10.0.0.5            save an SSH host
  {prog} history                                     saved runs
  {prog} generate busy --users 4                     make a narration workload
  {prog} calibrate 1                                 fit a model to a live run
  {prog} gui

`{prog} run -h` explains each flag. `{prog} gui` opens the desktop window.
With no terminal to answer, `{prog}`
compares every sample workload on every sample scheduler.
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
