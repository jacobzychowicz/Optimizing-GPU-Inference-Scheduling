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
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise ValueError(f"cannot load scheduler {path}: {exc}") from exc
    select = getattr(module, "select", None)
    if not callable(select):
        raise ValueError(f"{path} must define select(ready, now_ms)")
    return select


def duration_ms(model, job):
    """Real runtime for one job. Schedulers do not see this number."""
    runtime = model["runtime"]
    base = runtime["base_ms"]
    payload = job["input"]
    job_id = job.get("id", "?")
    if "per_character_ms" in runtime:
        field = runtime.get("text_field", "text")
        try:
            size = len(payload[field])
        except KeyError as exc:
            raise ValueError(f"job {job_id} is missing input.{field}") from exc
        except TypeError as exc:
            raise ValueError(f"job {job_id} input.{field} is not text") from exc
        return base + runtime["per_character_ms"] * size
    if "per_token_ms" in runtime:
        field = runtime.get("token_field", "max_tokens")
        try:
            size = payload[field]
        except KeyError as exc:
            raise ValueError(f"job {job_id} is missing input.{field}") from exc
        return base + runtime["per_token_ms"] * size
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
                "arrival_ms": arrival,
                "start_ms": start,
                "latency_ms": finish - arrival,
                "wait_ms": start - arrival,
                "finish_ms": finish,
                "service_ms": service,
                **job_tags(choice),
            }
        )
        now = finish

    return finished


def job_tags(job):
    """The optional user and kind of a job, copied onto its result so reports can group by them."""
    return {key: job[key] for key in ("kind", "user") if job.get(key) is not None}


def summarize(finished):
    """Mean latency, nearest-rank p95 latency, mean wait, and jobs per second.

    When jobs have a "kind" (for example script_segment and retake), by_kind holds
    the same latency numbers for each kind, since one average hides who waited.
    """
    count = len(finished)
    makespan_ms = max(job["finish_ms"] for job in finished)
    jobs_per_s = 0.0 if makespan_ms <= 0 else count / (makespan_ms / 1000)
    summary = _latency_stats(finished)
    summary["jobs_per_s"] = jobs_per_s
    kinds = {}
    for job in finished:
        if "kind" in job:
            kinds.setdefault(str(job["kind"]), []).append(job)
    if kinds:
        summary["by_kind"] = {kind: dict(_latency_stats(jobs), count=len(jobs)) for kind, jobs in sorted(kinds.items())}
    return summary


def _latency_stats(finished):
    latencies = [job["latency_ms"] for job in finished]
    waits = [job["wait_ms"] for job in finished]
    return {
        "mean_latency_ms": sum(latencies) / len(latencies),
        "p95_latency_ms": _nearest_rank(latencies, 0.95),
        "mean_wait_ms": sum(waits) / len(waits),
    }


def _nearest_rank(values, fraction):
    ordered = sorted(values)
    index = math.ceil(fraction * len(ordered)) - 1
    index = min(max(index, 0), len(ordered) - 1)
    return ordered[index]


def model_for(workload, override):
    if override is not None:
        return Path(override)
    name = workload.get("model")
    if not name:
        raise ValueError(f"workload {workload.get('name', '')} has no model; pass --model")
    from gpu_bench.store import user_dir

    for folder in (MODELS, user_dir("models")):
        candidate = folder / f"{name}.json"
        if candidate.is_file():
            return candidate
    return MODELS / f"{name}.json"


def check_jobs(workload, label):
    """Raise ValueError when a workload has no jobs or a job lacks a required field."""
    jobs = workload.get("jobs") or []
    if not jobs:
        raise ValueError(f"workload {label} has no jobs")
    seen = set()
    for job in jobs:
        missing = [key for key in ("id", "arrival_ms", "input") if key not in job]
        if missing:
            raise ValueError(f"job {job.get('id', '?')} in {label} is missing {', '.join(missing)}")
        if job["id"] in seen:
            raise ValueError(f"job id {job['id']} appears twice in {label}")
        seen.add(job["id"])
    return jobs


def comparison(workload_path, model_path, schedulers):
    """Run one workload and return the workload, model, and per-scheduler results.

    schedulers is a list of (display name, path) pairs, in the order to report.
    Each result has name, summary, and jobs. Each job has start_ms and finish_ms.
    """
    workload = load_json(workload_path)
    model = load_json(model_path)
    jobs = check_jobs(workload, workload.get("name") or workload_path)
    results = []
    for name, path in schedulers:
        select = load_select(path)
        finished = run_schedule(jobs, select, model, name)
        results.append({"name": name, "summary": summarize(finished), "jobs": finished})
    return {"workload": workload, "model": model, "results": results}


def compare(workload_path, model_path, schedulers):
    """Return the text report for one workload.

    schedulers is a list of (display name, path) pairs, in the order to print.
    """
    data = comparison(workload_path, model_path, schedulers)
    rows = [(item["name"], item["summary"]) for item in data["results"]]
    return format_report(data["workload"], data["model"], rows)


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
    if len(rows) > 1:
        mean, mean_names = _lowest(rows, "mean_latency_ms")
        p95, p95_names = _lowest(rows, "p95_latency_ms")
        lines.append(f"lowest mean latency: {', '.join(mean_names)} ({mean:.1f} ms)")
        lines.append(f"lowest p95 latency: {', '.join(p95_names)} ({p95:.1f} ms)")
    return "\n".join(lines)


def _lowest(rows, key):
    scored = [(round(summary[key], 1), name) for name, summary in rows]
    best = min(score for score, _name in scored)
    return best, [name for score, name in scored if score == best]
