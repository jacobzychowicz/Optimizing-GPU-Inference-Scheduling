"""Files the app keeps between runs: saved SSH hosts, run history, and settings.

Everything lives in ~/.gpu_bench unless GPU_BENCH_HOME points somewhere else.
"""

import csv
import json
import os
import re
import time
from pathlib import Path


SAFE_NAME = re.compile(r"^[A-Za-z0-9._-]+$")
# A host or user must never start with "-", or ssh would read it as an option.
SAFE_HOST = re.compile(r"^[A-Za-z0-9_.:\[\]%-]+$")
SAFE_USER = re.compile(r"^[A-Za-z0-9._-]+$")


def home():
    override = os.environ.get("GPU_BENCH_HOME")
    return Path(override).expanduser() if override else Path.home() / ".gpu_bench"


def user_dir(kind):
    """Folder for your own workloads, models, or schedulers, next to the samples."""
    return home() / kind


def _read(path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return default
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {path}: {exc}") from exc


def _write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(tmp, path)


# settings -----------------------------------------------------------------


def load_settings():
    data = _read(home() / "settings.json", {})
    return data if isinstance(data, dict) else {}


def save_settings(settings):
    _write(home() / "settings.json", settings)


# hosts --------------------------------------------------------------------


def normalize_host(raw):
    """Check one saved host and fill in defaults. Raises ValueError on bad input."""
    host = {
        "name": str(raw.get("name") or "").strip(),
        "host": str(raw.get("host") or "").strip(),
        "user": str(raw.get("user") or "").strip(),
        "port": raw.get("port") or None,
        "key": str(raw.get("key") or "").strip(),
        "python": str(raw.get("python") or "python3").strip() or "python3",
        "workdir": str(raw.get("workdir") or ".gpu-bench").strip() or ".gpu-bench",
        "ssh_options": [str(opt).strip() for opt in (raw.get("ssh_options") or []) if str(opt).strip()],
    }
    if not host["host"]:
        raise ValueError("a host needs a hostname or address")
    if host["host"].startswith("-") or not SAFE_HOST.match(host["host"]):
        raise ValueError(f"'{host['host']}' is not a valid hostname")
    if host["user"] and (host["user"].startswith("-") or not SAFE_USER.match(host["user"])):
        raise ValueError(f"'{host['user']}' is not a valid user name")
    if not host["name"]:
        host["name"] = host["host"]
    if not SAFE_NAME.match(host["name"]):
        raise ValueError("a host name may use letters, digits, dot, dash, and underscore")
    if host["port"] is not None:
        try:
            host["port"] = int(host["port"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"port '{host['port']}' is not a number") from exc
        if not 0 < host["port"] < 65536:
            raise ValueError(f"port {host['port']} is out of range")
    for opt in host["ssh_options"]:
        if opt.startswith("-") or "=" not in opt:
            raise ValueError(f"ssh option '{opt}' should look like Key=Value")
    if host["workdir"].startswith("~/"):
        host["workdir"] = host["workdir"][2:] or ".gpu-bench"
    return host


def load_hosts():
    data = _read(home() / "hosts.json", {"hosts": []})
    hosts = []
    for raw in data.get("hosts", []) if isinstance(data, dict) else []:
        try:
            hosts.append(normalize_host(raw))
        except ValueError:
            continue
    return hosts


def save_host(raw, replace=None):
    """Add or update a host. replace is the old name when renaming."""
    host = normalize_host(raw)
    hosts = [item for item in load_hosts() if item["name"] not in {host["name"], replace}]
    hosts.append(host)
    hosts.sort(key=lambda item: item["name"].lower())
    _write(home() / "hosts.json", {"hosts": hosts})
    return host


def remove_host(name):
    hosts = load_hosts()
    kept = [item for item in hosts if item["name"] != name]
    if len(kept) == len(hosts):
        raise ValueError(f"no saved host named '{name}'")
    _write(home() / "hosts.json", {"hosts": kept})


def parse_target(text):
    """Turn user@host:port into a host dict, without saving it."""
    text = text.strip()
    user, _, rest = text.rpartition("@")
    port = None
    if rest.count(":") == 1:
        rest, port = rest.split(":")
    return normalize_host({"name": re.sub(r"[^A-Za-z0-9._-]", "_", text), "host": rest, "user": user, "port": port})


def find_host(token):
    """A saved host by name, or an ad-hoc user@host[:port]."""
    for host in load_hosts():
        if host["name"] == token:
            return host
    if "@" in token or "." in token or ":" in token:
        return parse_target(token)
    names = ", ".join(host["name"] for host in load_hosts()) or "(none saved)"
    raise ValueError(f"no saved host named '{token}'. Saved hosts: {names}. Or pass user@host.")


def describe_host(host):
    dest = f"{host['user']}@{host['host']}" if host.get("user") else host["host"]
    if host.get("port"):
        dest += f":{host['port']}"
    return dest


# run history --------------------------------------------------------------


def runs_dir():
    return home() / "runs"


def save_run(result):
    folder = runs_dir()
    folder.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    slug = re.sub(r"[^A-Za-z0-9._-]", "_", str(result.get("workload", {}).get("name") or "run"))[:40]
    path = folder / f"{stamp}-{slug}.json"
    count = 1
    while path.exists():
        count += 1
        path = folder / f"{stamp}-{slug}-{count}.json"
    _write(path, result)
    return path


def list_runs():
    """Saved runs, newest first, with enough detail to show in a list."""
    rows = []
    for path in sorted(runs_dir().glob("*.json"), reverse=True):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        results = data.get("results") or []
        best = ""
        if results:
            top = min(results, key=lambda item: item["summary"]["mean_latency_ms"])
            best = top["name"]
        rows.append(
            {
                "path": path,
                "created": data.get("created", ""),
                "workload": data.get("workload", {}).get("name", "?"),
                "model": data.get("model", {}).get("name", "?"),
                "mode": data.get("mode", "simulate"),
                "target": data.get("target", "local"),
                "schedulers": [item["name"] for item in results],
                "best": best,
            }
        )
    return rows


def load_run(path):
    return _read(Path(path), None)


def delete_run(path):
    Path(path).unlink()


SUMMARY_COLUMNS = (
    ("scheduler", None),
    ("mean_ms", "mean_latency_ms"),
    ("p95_ms", "p95_latency_ms"),
    ("wait_ms", "mean_wait_ms"),
    ("jobs_per_s", "jobs_per_s"),
    ("gpu_util_pct", "gpu_util_pct"),
    ("gpu_mem_peak_mb", "gpu_mem_peak_mb"),
)
JOB_COLUMNS = ("scheduler", "id", "arrival_ms", "start_ms", "finish_ms", "wait_ms", "latency_ms", "service_ms")


def export(result, path, kind="summary"):
    """Write a run to .json, or to .csv (one row per scheduler, or per job with kind='jobs')."""
    path = Path(path)
    if path.suffix.lower() == ".json":
        path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return path
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        context = [result.get("workload", {}).get("name"), result.get("model", {}).get("name"),
                   result.get("mode"), result.get("target"), result.get("created")]
        if kind == "jobs":
            writer.writerow([*JOB_COLUMNS, "workload", "model", "mode", "target", "created"])
            for item in result.get("results", []):
                for job in item.get("jobs", []):
                    writer.writerow([item["name"], *(job.get(col, "") for col in JOB_COLUMNS[1:]), *context])
        else:
            writer.writerow([label for label, _key in SUMMARY_COLUMNS] + ["workload", "model", "mode", "target", "created"])
            for item in result.get("results", []):
                summary = item["summary"]
                row = [item["name"]] + [_fmt(summary.get(key)) for _label, key in SUMMARY_COLUMNS[1:]]
                writer.writerow(row + context)
    return path


def _fmt(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.3f}"
    return value
