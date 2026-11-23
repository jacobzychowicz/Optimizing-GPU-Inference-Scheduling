"""Run one workload through several schedulers, simulated or live, and report.

The command line, the menu, the desktop window, and SSH runs all go through
execute(). Progress is reported as small event dicts so the same run can be
printed, streamed over SSH as JSON lines, or drawn in the window.

Events:
  {"event": "start", "workload", "model", "mode", "schedulers", "machine"}
  {"event": "log", "message"}
  {"event": "scheduler_start", "scheduler", "repeat", "repeats"}
  {"event": "job_start", "scheduler", "repeat", "id", "start_ms", "done", "total"}   (live only)
  {"event": "job", "scheduler", "repeat", "done", "total", "job": {...}}
  {"event": "scheduler_done", "scheduler", "summary"}
  {"event": "result", "data": {...}}   (always last)
"""

import platform
import socket
import threading
import time

from gpu_bench.compare import check_jobs, job_tags, load_json, load_select, run_schedule, summarize


MODES = ("simulate", "live")


class Cancelled(RuntimeError):
    """The user stopped the run."""


def execute(workload_path, model_path, schedulers, mode="simulate", repeats=1, emit=None, cancel=None):
    """Run every scheduler and return the result dict.

    schedulers: list of (display name, path).
    cancel: a threading.Event; setting it stops a live run between jobs.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {', '.join(MODES)}")
    emit = emit or (lambda _event: None)
    cancel = cancel or threading.Event()
    repeats = max(1, int(repeats))
    if mode == "simulate":
        repeats = 1  # a simulation gives the same answer every time

    workload = load_json(workload_path)
    model = load_json(model_path)
    jobs = check_jobs(workload, workload.get("name") or workload_path)
    workload.setdefault("name", str(workload_path))
    model.setdefault("name", str(model_path))
    loaded = [(name, load_select(path)) for name, path in schedulers]
    if not loaded:
        raise ValueError("pick at least one scheduler")

    result = {
        "version": 1,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "mode": mode,
        "target": "local",
        "machine": socket.gethostname(),
        "python": platform.python_version(),
        "workload": workload,
        "model": model,
        "repeats": repeats,
        "gpus": [],
        "results": [],
    }
    emit(
        {
            "event": "start",
            "workload": workload["name"],
            "model": model["name"],
            "mode": mode,
            "schedulers": [name for name, _select in loaded],
            "machine": result["machine"],
        }
    )

    if mode == "simulate":
        from gpu_bench.backends import can_simulate

        if not can_simulate(model):
            raise ValueError(
                f"{model['name']} has no runtime formula to simulate. Run it live instead (--mode live)."
            )
        for name, select in loaded:
            emit({"event": "scheduler_start", "scheduler": name, "repeat": 1, "repeats": 1})
            finished = run_schedule(jobs, select, model, name)
            summary = summarize(finished)
            result["results"].append({"name": name, "summary": summary, "jobs": finished, "runs": [summary]})
            emit({"event": "scheduler_done", "scheduler": name, "summary": summary})
    else:
        _execute_live(result, jobs, model, model_path, loaded, repeats, emit, cancel)

    emit({"event": "result", "data": result})
    return result


def _execute_live(result, jobs, model, model_path, loaded, repeats, emit, cancel):
    from gpu_bench.backends import describe, open_backend
    from gpu_bench.gpus import Sampler, query_nvidia

    result["gpus"] = query_nvidia()
    if result["gpus"]:
        names = ", ".join(gpu["name"] for gpu in result["gpus"])
        emit({"event": "log", "message": f"GPUs on {result['machine']}: {names}"})
    else:
        emit({"event": "log", "message": f"No NVIDIA GPU found on {result['machine']}; GPU use will not be sampled."})
    emit({"event": "log", "message": f"Loading {model['name']} ({describe(model)})"})
    backend = open_backend(model, model_path)
    try:
        emit({"event": "log", "message": "Warming up"})
        backend.warmup()
        for name, select in loaded:
            runs = []
            last = []
            for repeat in range(1, repeats + 1):
                if cancel.is_set():
                    raise Cancelled("stopped")
                emit({"event": "scheduler_start", "scheduler": name, "repeat": repeat, "repeats": repeats})
                sampler = Sampler().start()
                try:
                    finished = run_live(jobs, select, backend, name, cancel, emit, repeat)
                finally:
                    stats = sampler.stop()
                summary = summarize(finished)
                summary.update(stats)
                runs.append(summary)
                last = finished
            summary = _average(runs)
            result["results"].append({"name": name, "summary": summary, "jobs": last, "runs": runs})
            emit({"event": "scheduler_done", "scheduler": name, "summary": summary})
    finally:
        backend.close()


def run_live(jobs, select, backend, scheduler_name, cancel, emit, repeat=1):
    """Replay the workload on a real clock. Requests arrive at arrival_ms after the start.

    Non-preemptive like the simulation: the backend finishes one request before
    the scheduler chooses the next.
    """
    pending = sorted((dict(job) for job in jobs), key=lambda job: (job["arrival_ms"], job["id"]))
    total = len(pending)
    ready = []
    finished = []
    start_clock = time.perf_counter()

    def elapsed_ms():
        return (time.perf_counter() - start_clock) * 1000.0

    while pending or ready:
        if cancel.is_set():
            raise Cancelled("stopped")
        now = elapsed_ms()
        while pending and pending[0]["arrival_ms"] <= now:
            ready.append(pending.pop(0))
        if not ready:
            wait_s = (pending[0]["arrival_ms"] - now) / 1000.0
            cancel.wait(min(max(wait_s, 0.0), 0.25))
            continue

        try:
            choice = select(list(ready), now)
        except Exception as exc:
            raise RuntimeError(f"{scheduler_name} failed at {now:.0f} ms: {exc}") from exc
        if not any(choice is job for job in ready):
            raise RuntimeError(f"{scheduler_name} returned a job that is not waiting")
        ready = [job for job in ready if job is not choice]

        start = elapsed_ms()
        emit(
            {
                "event": "job_start",
                "scheduler": scheduler_name,
                "repeat": repeat,
                "id": choice["id"],
                "start_ms": round(start, 3),
                "done": len(finished),
                "total": total,
            }
        )
        backend.run(choice)
        finish = elapsed_ms()
        arrival = choice["arrival_ms"]
        record = {
            "id": choice["id"],
            "arrival_ms": arrival,
            "start_ms": round(start, 3),
            "latency_ms": round(finish - arrival, 3),
            "wait_ms": round(start - arrival, 3),
            "finish_ms": round(finish, 3),
            "service_ms": round(finish - start, 3),
            **job_tags(choice),
        }
        finished.append(record)
        emit(
            {
                "event": "job",
                "scheduler": scheduler_name,
                "repeat": repeat,
                "done": len(finished),
                "total": total,
                "job": record,
            }
        )
    return finished


def _average(runs):
    if len(runs) == 1:
        return dict(runs[0])
    keys = set()
    for run in runs:
        keys.update(run)
    keys.discard("by_kind")
    average = {}
    for key in keys:
        values = [run[key] for run in runs if key in run]
        average[key] = max(values) if key == "gpu_mem_peak_mb" else sum(values) / len(values)
    kinds = {}
    for run in runs:
        for kind, stats in (run.get("by_kind") or {}).items():
            kinds.setdefault(kind, []).append(stats)
    if kinds:
        average["by_kind"] = {
            kind: {key: sum(stats[key] for stats in group) / len(group) for key in group[0]}
            for kind, group in kinds.items()
        }
    means = [run["mean_latency_ms"] for run in runs]
    center = sum(means) / len(means)
    average["mean_latency_spread_ms"] = (sum((value - center) ** 2 for value in means) / (len(means) - 1)) ** 0.5
    return average


def format_result(result):
    """Plain-text table, the same shape the command line has always printed."""
    rows = result["results"]
    workload, model = result["workload"], result["model"]
    width = max([20, *(len(item["name"]) for item in rows)])
    gpu_cols = any("gpu_util_pct" in item["summary"] for item in rows)
    lines = [
        f"workload: {workload['name']}",
        f"model: {model['name']}",
        f"jobs: {len(workload['jobs'])}",
    ]
    if result.get("mode", "simulate") != "simulate" or result.get("target", "local") != "local":
        where = result.get("target", "local")
        machine = result.get("machine")
        lines.append(f"mode: {result.get('mode')}  on: {where}" + (f" ({machine})" if machine and machine != where else ""))
        if result.get("gpus"):
            lines.append("gpus: " + ", ".join(gpu["name"] for gpu in result["gpus"]))
        if result.get("repeats", 1) > 1:
            lines.append(f"repeats: {result['repeats']} (numbers are averages)")
    header = f"{'scheduler':<{width}} {'mean_ms':>10} {'p95_ms':>10} {'wait_ms':>10} {'jobs/s':>8}"
    if gpu_cols:
        header += f" {'gpu_%':>6} {'mem_mb':>8}"
    lines += ["", header]
    for item in rows:
        summary = item["summary"]
        line = (
            f"{item['name']:<{width}} "
            f"{summary['mean_latency_ms']:10.1f} "
            f"{summary['p95_latency_ms']:10.1f} "
            f"{summary['mean_wait_ms']:10.1f} "
            f"{summary['jobs_per_s']:8.2f}"
        )
        if gpu_cols:
            util = summary.get("gpu_util_pct")
            mem = summary.get("gpu_mem_peak_mb")
            line += f" {util:6.0f}" if util is not None else f" {'-':>6}"
            line += f" {mem:8.0f}" if mem is not None else f" {'-':>8}"
        lines.append(line)
        for kind, stats in (summary.get("by_kind") or {}).items():
            label = f"  {kind} ({stats['count']:.0f})"
            lines.append(
                f"{label:<{width}} "
                f"{stats['mean_latency_ms']:10.1f} "
                f"{stats['p95_latency_ms']:10.1f} "
                f"{stats['mean_wait_ms']:10.1f}"
            )
    lines.append("")
    lines.append("mean_ms and p95_ms are latency from arrival to finish. wait_ms is time spent waiting.")
    if len(rows) > 1:
        for key, label in (("mean_latency_ms", "mean latency"), ("p95_latency_ms", "p95 latency")):
            best, names = lowest(rows, key)
            lines.append(f"lowest {label}: {', '.join(names)} ({best:.1f} ms)")
    return "\n".join(lines)


def lowest(rows, key):
    scored = [(round(item["summary"][key], 1), item["name"]) for item in rows]
    best = min(score for score, _name in scored)
    return best, [name for score, name in scored if score == best]
