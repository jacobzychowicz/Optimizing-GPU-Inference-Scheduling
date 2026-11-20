"""Load workloads, models, and schedulers, then compare one run of each policy."""

import copy
import importlib.util
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKLOADS = ROOT / "defaults" / "workloads"
MODELS = ROOT / "defaults" / "models"
SCHEDULERS = ROOT / "defaults" / "schedulers"


def load_json(path):
    """Read one JSON file. The error names the file."""
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"cannot read {path}: {exc.strerror}") from exc
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON: {exc}") from exc


def load_select(path):
    """Import a scheduler file and return its select function."""
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"cannot read scheduler {path}")
    spec = importlib.util.spec_from_file_location(f"scheduler_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load scheduler {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    select = getattr(module, "select", None)
    if not callable(select):
        raise ValueError(f"{path} must define select(ready, now_ms)")
    return select


def duration_ms(model, job):
    """Real runtime for one job. Schedulers do not see this number."""
    runtime = model["runtime"]
    base = runtime["base_ms"]
    payload = job["input"]
    if "per_character_ms" in runtime:
        field = runtime.get("text_field", "text")
        return base + runtime["per_character_ms"] * len(payload[field])
    if "per_token_ms" in runtime:
        field = runtime.get("token_field", "max_tokens")
        return base + runtime["per_token_ms"] * payload[field]
    name = model.get("name", "model")
    raise ValueError(f"{name} has no per_character_ms or per_token_ms runtime")


def run_schedule(jobs, select, model, scheduler_name):
    """Run one non-preemptive schedule. The GPU finishes a job before the next choice."""
    pending = sorted(copy.deepcopy(jobs), key=lambda job: (job["arrival_ms"], job["id"]))
    ready = []
    finished = []
    now = 0

    while pending or ready:
        if not ready:
            now = max(now, pending[0]["arrival_ms"])
        while pending and pending[0]["arrival_ms"] <= now:
            ready.append(pending.pop(0))

        try:
            choice = select(list(ready), now)
        except Exception as exc:
            raise RuntimeError(f"{scheduler_name} failed at {now} ms: {exc}") from exc

        if not any(choice is job for job in ready):
            raise RuntimeError(f"{scheduler_name} returned a job that is not waiting")

        ready = [job for job in ready if job is not choice]
        service = duration_ms(model, choice)
        if service < 0:
            raise RuntimeError(f"{scheduler_name} got a negative runtime for {choice['id']}")

        start = now
        finish = now + service
        arrival = choice["arrival_ms"]
        finished.append(
            {
                "id": choice["id"],
                "latency_ms": finish - arrival,
                "wait_ms": start - arrival,
                "finish_ms": finish,
            }
        )
        now = finish

    return finished


def summarize(finished):
    """Mean latency, nearest-rank p95 latency, mean wait, and jobs per second."""
    latencies = [job["latency_ms"] for job in finished]
    waits = [job["wait_ms"] for job in finished]
    count = len(finished)
    makespan_ms = max(job["finish_ms"] for job in finished)
    jobs_per_s = 0.0 if makespan_ms <= 0 else count / (makespan_ms / 1000)
    return {
        "mean_latency_ms": sum(latencies) / count,
        "p95_latency_ms": _nearest_rank(latencies, 0.95),
        "mean_wait_ms": sum(waits) / count,
        "jobs_per_s": jobs_per_s,
    }


def _nearest_rank(values, fraction):
    ordered = sorted(values)
    index = math.ceil(fraction * len(ordered)) - 1
    index = min(max(index, 0), len(ordered) - 1)
    return ordered[index]


def default_workloads():
    return sorted(WORKLOADS.glob("*.json"))


def default_schedulers():
    return sorted(path for path in SCHEDULERS.glob("*.py") if not path.name.startswith("_"))


def model_for(workload, override):
    if override is not None:
        return Path(override)
    name = workload.get("model")
    if not name:
        raise ValueError(f"workload {workload.get('name', '')} has no model; pass --model")
    return MODELS / f"{name}.json"


def compare(workload_path, model_path, scheduler_paths):
    """Return the text report for one workload."""
    workload = load_json(workload_path)
    model = load_json(model_path)
    rows = []
    for path in scheduler_paths:
        select = load_select(path)
        finished = run_schedule(workload["jobs"], select, model, path.stem)
        rows.append((path.stem, summarize(finished)))
    return format_report(workload, model, rows)


def format_report(workload, model, rows):
    width = max([20, *(len(name) for name, _summary in rows)])
    lines = [
        f"workload: {workload['name']}",
        f"model: {model['name']}",
        f"jobs: {len(workload['jobs'])}",
        "",
        f"{'scheduler':<{width}} {'mean_ms':>10} {'p95_ms':>10} {'wait_ms':>10} {'jobs/s':>8}",
    ]
    for name, summary in rows:
        lines.append(
            f"{name:<{width}} "
            f"{summary['mean_latency_ms']:10.1f} "
            f"{summary['p95_latency_ms']:10.1f} "
            f"{summary['mean_wait_ms']:10.1f} "
            f"{summary['jobs_per_s']:8.2f}"
        )
    lines.append("")
    lines.append("mean_ms and p95_ms are latency from arrival to finish. wait_ms is time spent waiting.")
    return "\n".join(lines)
