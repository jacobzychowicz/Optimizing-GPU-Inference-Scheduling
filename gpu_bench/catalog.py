"""Names you can type instead of file paths."""

import ast
import difflib
from pathlib import Path

from gpu_bench.compare import MODELS, SCHEDULERS, WORKLOADS, load_json


def workload_items():
    items = []
    for path in sorted(WORKLOADS.glob("*.json")):
        data = load_json(path)
        items.append(
            {
                "kind": "workload",
                "name": path.stem,
                "path": path,
                "keys": _keys(path, data.get("name")),
                "description": first_sentence(data.get("description", "")),
                "model": data.get("model") or "",
            }
        )
    return items


def model_items():
    items = []
    for path in sorted(MODELS.glob("*.json")):
        data = load_json(path)
        items.append(
            {
                "kind": "model",
                "name": path.stem,
                "path": path,
                "keys": _keys(path, data.get("name")),
                "description": first_sentence(data.get("description", "")),
                "model": "",
            }
        )
    return items


def scheduler_items():
    items = []
    for path in sorted(path for path in SCHEDULERS.glob("*.py") if not path.name.startswith("_")):
        items.append(
            {
                "kind": "scheduler",
                "name": path.stem,
                "path": path,
                "keys": _keys(path, None),
                "description": module_summary(path),
                "model": "",
            }
        )
    return items


def resolve(token, kind=None):
    """Find a workload, model, or scheduler by name or path.

    kind limits the search to one of those. None accepts any of the three.
    """
    path = Path(token)
    if path.exists() and path.is_dir():
        raise ValueError(f"{token} is a directory. Pass a file, or a name from `gpu-bench list`.")
    if path.is_file():
        item = item_from_file(path)
        if kind and item["kind"] != kind:
            raise ValueError(f"{token} is a {item['kind']}, not a {kind}")
        return item

    matches = _lookup(token, kind)
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        flags = " or ".join(f"--{item['kind']}" for item in matches)
        raise ValueError(f"'{token}' matches more than one name. Pass {flags}.")
    raise ValueError(_unknown(token, kind))


def item_from_file(path):
    path = Path(path)
    if path.suffix == ".py":
        return {
            "kind": "scheduler",
            "name": path.stem,
            "path": path,
            "keys": _keys(path, None),
            "description": module_summary(path),
            "model": "",
        }
    if path.suffix == ".json":
        data = load_json(path)
        if "jobs" in data and "runtime" in data:
            raise ValueError(f"{path} looks like both a workload and a model")
        if "jobs" in data:
            return {
                "kind": "workload",
                "name": data.get("name") or path.stem,
                "path": path,
                "keys": _keys(path, data.get("name")),
                "description": first_sentence(data.get("description", "")),
                "model": data.get("model") or "",
            }
        if "runtime" in data:
            return {
                "kind": "model",
                "name": data.get("name") or path.stem,
                "path": path,
                "keys": _keys(path, data.get("name")),
                "description": first_sentence(data.get("description", "")),
                "model": "",
            }
        raise ValueError(f"{path} is not a workload or a model")
    raise ValueError(f"{path} must be a .json workload or model, or a .py scheduler")


def format_catalog(kind=None):
    sections = []
    if kind in (None, "workloads"):
        rows = [(item["name"], item["model"] or "-", item["description"] or "-") for item in workload_items()]
        sections.append("workloads\n" + _table(("name", "model", "what it is"), rows))
    if kind in (None, "models"):
        rows = [(item["name"], item["description"] or "-") for item in model_items()]
        sections.append("models\n" + _table(("name", "what it is"), rows))
    if kind in (None, "schedulers"):
        rows = [(item["name"], item["description"] or "-") for item in scheduler_items()]
        sections.append("schedulers\n" + _table(("name", "what it does"), rows))
    return "\n\n".join(sections)


def first_sentence(text):
    paragraph = text.strip().split("\n\n", 1)[0]
    paragraph = " ".join(paragraph.split())
    if not paragraph:
        return ""
    for index, char in enumerate(paragraph):
        if char == "." and (index + 1 == len(paragraph) or paragraph[index + 1] == " "):
            return paragraph[: index + 1]
    return paragraph


def module_summary(path):
    try:
        tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return "could not read this file"
    return first_sentence(ast.get_docstring(tree) or "")


def _keys(path, extra):
    keys = {path.stem}
    if extra:
        keys.add(extra)
    return keys


def _catalogs(kind):
    catalogs = []
    if kind in (None, "workload"):
        catalogs.append(workload_items())
    if kind in (None, "model"):
        catalogs.append(model_items())
    if kind in (None, "scheduler"):
        catalogs.append(scheduler_items())
    return catalogs


def _lookup(token, kind):
    stem = Path(token).stem if Path(token).suffix in {".json", ".py"} else token
    matches = []
    for items in _catalogs(kind):
        for item in items:
            if token in item["keys"] or stem in item["keys"]:
                matches.append(item)
                break
    return matches


def _unknown(token, kind):
    choices = [item["name"] for items in _catalogs(kind) for item in items]
    suggestion = difflib.get_close_matches(token, choices, n=1, cutoff=0.6)
    hint = f" Did you mean {suggestion[0]}?" if suggestion else ""
    if kind:
        available = ", ".join(choices) if choices else "(none)"
        return f"unknown {kind} '{token}'.{hint}\navailable: {available}"

    lines = [f"no workload, model, or scheduler named '{token}'.{hint}"]
    for label, items in (
        ("workloads", workload_items()),
        ("models", model_items()),
        ("schedulers", scheduler_items()),
    ):
        names = ", ".join(item["name"] for item in items) or "(none)"
        lines.append(f"{label}: {names}")
    return "\n".join(lines)


def _table(headers, rows):
    if not rows:
        return "  (none)"
    widths = [len(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    def line(cells):
        return "  ".join(cell.ljust(widths[index]) for index, cell in enumerate(cells)).rstrip()

    body = [line(headers)]
    body.extend(line(row) for row in rows)
    return "\n".join("  " + row for row in body)
