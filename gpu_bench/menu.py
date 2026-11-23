"""Numbered menu. Each choice opens a list of files loaded from defaults/."""

import shlex
import sys

from gpu_bench.catalog import model_items, scheduler_items, workload_items
from gpu_bench.compare import MODELS, ROOT, SCHEDULERS, WORKLOADS


WIDTH = 72


class _Quit(Exception):
    """The user asked to leave the menu."""


def run_menu(read_line=None):
    """Prompt until the user quits. read_line defaults to input()."""
    if read_line is None:
        read_line = input
    _enable_ansi()
    state = {"model": None, "workload": None, "schedulers": [], "mode": "simulate", "host": None}
    note = "Change any choice before you run."
    while True:
        try:
            _clear()
            print(_main_screen(state, note))
            choice = read_line("Choose: ").strip().lower()
            if choice == "1":
                picked = _pick_one(
                    "Select a model",
                    _folder_label(MODELS),
                    model_items(),
                    state["model"],
                    "Choose a model: ",
                    read_line,
                )
                if picked is not None:
                    state["model"] = picked
                    note = f"Selected model: {picked['name']}"
            elif choice == "2":
                picked = _pick_one(
                    "Select a workload",
                    _folder_label(WORKLOADS),
                    workload_items(),
                    state["workload"],
                    "Choose a workload: ",
                    read_line,
                )
                if picked is not None:
                    state["workload"] = picked
                    note = f"Selected workload: {picked['name']}"
            elif choice == "3":
                state["schedulers"] = _pick_many(
                    "Select schedulers",
                    _folder_label(SCHEDULERS),
                    scheduler_items(),
                    state["schedulers"],
                    read_line,
                )
                names = _names(state["schedulers"])
                note = f"Selected schedulers: {names or '(none)'}"
            elif choice == "4":
                note = _execute(state, read_line)
            elif choice == "5":
                note = _legacy(read_line)
            elif choice == "6":
                state["mode"] = "live" if state["mode"] == "simulate" else "simulate"
                note = f"Mode: {_mode_label(state['mode'])}"
            elif choice == "7":
                note = _pick_host(state, read_line)
            elif choice in {"q", "quit", "exit"}:
                return 0
            elif choice == "":
                note = "Type 1 to 7, or q."
            else:
                note = "Type 1 for model, 2 for workload, 3 for schedulers, 4 to run, 5 for the typed command line, 6 for mode, 7 for machine, or q to quit."
        except _Quit:
            return 0
        except EOFError:
            print()
            return 0
        except KeyboardInterrupt:
            print()
            return 130


def _pick_one(title, folder, items, current, prompt, read_line):
    selected = {current["name"]} if current else set()
    note = "Current: " + (current["name"] if current else "(none)")
    while True:
        _clear()
        print(_list_screen(title, folder, items, selected, note, many=False))
        if not items:
            raw = read_line("Choose: ").strip().lower()
            if raw in {"b", "back", "q", "quit", "exit"}:
                if raw in {"q", "quit", "exit"}:
                    raise _Quit()
                return None
            note = f"No files in {folder}. Add one, then open this list again."
            continue
        raw = read_line(prompt).strip().lower()
        if raw in {"b", "back"}:
            return None
        if raw in {"q", "quit", "exit"}:
            raise _Quit()
        index = _number(raw, len(items))
        if index is None:
            note = f"Type a number from 1 to {len(items)}, or b to go back."
            continue
        return items[index]


def _pick_many(title, folder, items, current, read_line):
    selected = {item["name"] for item in current}
    note = "Type a number to add or remove it."
    while True:
        chosen = [item for item in items if item["name"] in selected]
        _clear()
        print(_list_screen(title, folder, items, selected, note, many=True, chosen=chosen))
        raw = read_line("Choose schedulers: ").strip().lower()
        if raw in {"b", "back"}:
            return chosen
        if raw in {"q", "quit", "exit"}:
            raise _Quit()
        if not items:
            note = f"No files in {folder}. Add one, then open this list again."
            continue
        if raw == "a":
            selected = {item["name"] for item in items}
            note = "Selected every scheduler."
            continue
        if raw == "c":
            selected = set()
            note = "Cleared the schedulers."
            continue
        index = _number(raw, len(items))
        if index is None:
            note = f"Type a number from 1 to {len(items)}, a for all, c to clear, or b to go back."
            continue
        name = items[index]["name"]
        if name in selected:
            selected.remove(name)
            note = f"Removed {name}."
        else:
            selected.add(name)
            note = f"Selected {name}."


def _execute(state, read_line):
    missing = []
    if state["model"] is None:
        missing.append("a model")
    if state["workload"] is None:
        missing.append("a workload")
    if not state["schedulers"]:
        missing.append("at least one scheduler")
    if missing:
        return "Still needed before running: " + ", ".join(missing) + "."

    for label, item in (
        ("model", state["model"]),
        ("workload", state["workload"]),
    ):
        if not item["path"].is_file():
            return f"The {label} file is no longer in the folder. Choose it again."
    for item in state["schedulers"]:
        if not item["path"].is_file():
            return f"The scheduler file {item['name']} is no longer in the folder. Choose schedulers again."

    from gpu_bench.cli import _progress_printer, run_anywhere
    from gpu_bench.runner import format_result

    _clear()
    chosen = [(item["name"], item["path"]) for item in state["schedulers"]]
    try:
        result = run_anywhere(
            state["host"],
            state["workload"]["path"],
            state["model"]["path"],
            chosen,
            state["mode"],
            1,
            _progress_printer(state["mode"], state["host"]),
        )
        print(format_result(result))
    except KeyboardInterrupt:
        print()
        print("stopped")
        note = "Run stopped."
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        note = f"Run failed: {exc}"
    else:
        names = _names(state["schedulers"])
        note = f"Last run: {state['workload']['name']} on {state['model']['name']} with {names}."
    read_line("Press Enter to return to the menu.")
    return note


def _legacy(read_line):
    from gpu_bench.catalog import format_catalog
    from gpu_bench.cli import cmd_run, dispatch

    _clear()
    print(format_catalog())
    print()
    print("Legacy command line. Type a command the way you would after python -m gpu_bench.")
    print("`all` runs every sample. An empty line returns to the menu.")
    print("Examples: mixed-lengths fifo priority")
    print("          token-mix --model example-llm")
    while True:
        line = read_line("legacy> ").strip()
        if line == "":
            return "Back at the menu."
        if line in {"menu", "back"}:
            return "Back at the menu."
        if line in {"q", "quit", "exit"}:
            raise _Quit()
        try:
            if line == "all":
                cmd_run([])
            else:
                args = shlex.split(line)
                if args and args[0] == "menu":
                    return "Back at the menu."
                dispatch(args)
        except (ValueError, RuntimeError, KeyError) as exc:
            print(f"error: {exc}", file=sys.stderr)
        except SystemExit as exc:
            if exc.code not in (0, None, 2):
                raise
        print()


def _main_screen(state, note):
    lines = [
        "Files loaded from defaults/",
        "",
        *_field("model", _names([state["model"]]) if state["model"] else "(none)"),
        *_field("workload", _names([state["workload"]]) if state["workload"] else "(none)"),
        *_field("schedulers", _names(state["schedulers"]) if state["schedulers"] else "(none)"),
        *_field("mode", _mode_label(state["mode"])),
        *_field("machine", _host_label(state["host"])),
        "",
        "1  model",
        "2  workload",
        "3  schedulers",
        "4  run",
        "5  legacy command line",
        "6  switch simulate / live",
        "7  machine (this one or an SSH host)",
        "q  quit",
        "",
        note,
    ]
    return _frame("GPU inference scheduler", lines)


def _mode_label(mode):
    if mode == "live":
        return "live (real clock, runs the model's backend)"
    return "simulate (computed from the runtime formula)"


def _host_label(host):
    if host is None:
        return "this machine"
    from gpu_bench.store import describe_host

    return f"{host['name']} ({describe_host(host)})"


def _pick_host(state, read_line):
    from gpu_bench.store import find_host, load_hosts

    hosts = load_hosts()
    note = "0 is this machine. Or type user@host to use a host you have not saved."
    while True:
        lines = ["Saved hosts come from `hosts add` or the desktop window.", "", " 0  this machine"]
        for index, host in enumerate(hosts, start=1):
            lines.append(f"{index:>2}  {_host_label(host)}")
        lines += ["", "b  back", "", note]
        _clear()
        print(_frame("Select a machine", lines))
        raw = read_line("Choose a machine: ").strip()
        if raw.lower() in {"b", "back", ""}:
            return f"Machine: {_host_label(state['host'])}"
        if raw.lower() in {"q", "quit", "exit"}:
            raise _Quit()
        if raw == "0":
            state["host"] = None
            return "Machine: this machine"
        index = _number(raw, len(hosts))
        try:
            state["host"] = hosts[index] if index is not None else find_host(raw)
        except ValueError as exc:
            note = str(exc)
            continue
        return f"Machine: {_host_label(state['host'])}"


def _list_screen(title, folder, items, selected, note, many, chosen=None):
    lines = [f"Loaded from {folder}", ""]
    if not items:
        lines.append(f"No files in {folder}.")
    for index, item in enumerate(items, start=1):
        mark = " *" if item["name"] in selected else ""
        lines.append(f"{index:>2}  {item['name']}{mark}")
        if item.get("model"):
            lines.append(f"    pairs with {item['model']}")
        for desc_line in _wrap(item.get("description") or "", WIDTH - 4):
            lines.append(f"    {desc_line}")
        lines.append("")
    if many:
        lines.append("Selected: " + (_names(chosen or []) or "(none)"))
        lines.append("")
        lines.extend(["a  all", "c  clear"])
    lines.append("b  back")
    lines.append("")
    lines.append(note)
    return _frame(title, lines)


def _frame(title, lines):
    width = max(WIDTH, len(title))
    body = []
    for line in lines:
        if len(line) <= width:
            body.append(line)
        else:
            body.extend(_wrap(line, width))
    bar = "+" + "-" * (width + 2) + "+"
    framed = [bar, "| " + title.ljust(width) + " |", bar]
    for line in body:
        framed.append("| " + line.ljust(width) + " |")
    framed.append(bar)
    return "\n".join(framed)


def _field(label, text):
    prefix = f"{label:<12}"
    room = WIDTH - len(prefix) - 1
    chunks = _wrap(text, room)
    lines = [f"{prefix} {chunks[0]}"]
    pad = " " * (len(prefix) + 1)
    for chunk in chunks[1:]:
        lines.append(pad + chunk)
    return lines


def _wrap(text, width):
    if text == "":
        return [""]
    words = text.split()
    if not words:
        return [""]
    lines = []
    current = ""
    for word in words:
        while len(word) > width:
            if current:
                lines.append(current)
                current = ""
            lines.append(word[:width])
            word = word[width:]
        trial = word if not current else f"{current} {word}"
        if len(trial) <= width:
            current = trial
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines or [""]


def _names(items):
    return ", ".join(item["name"] for item in items)


def _number(raw, count):
    if not raw.isdigit():
        return None
    value = int(raw)
    if 1 <= value <= count:
        return value - 1
    return None


def _folder_label(path):
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def _clear():
    if not sys.stdout.isatty():
        print()
        return
    sys.stdout.write("\033[2J\033[H")
    sys.stdout.flush()


def _enable_ansi():
    if sys.platform != "win32":
        return
    try:
        import ctypes

        handle = ctypes.windll.kernel32.GetStdHandle(-11)
        mode = ctypes.c_uint()
        if ctypes.windll.kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            ctypes.windll.kernel32.SetConsoleMode(handle, mode.value | 0x0004)
    except Exception:
        return
