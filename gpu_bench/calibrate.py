"""Fit a model's runtime formula to what a live run measured.

A live run records how long the GPU took for every request (service_ms). Fitting
base_ms + per_unit_ms * size to those times gives a simulator that behaves like
the real hardware, so policies can be compared in milliseconds instead of
re-running the GPU each time. check() re-simulates the same workload with the
fitted model and compares it to the live numbers.
"""

import copy


def _size_field(model, jobs):
    runtime = model.get("runtime") or {}
    if "per_character_ms" in runtime:
        return "characters", runtime.get("text_field", "text")
    if "per_token_ms" in runtime:
        return "tokens", runtime.get("token_field", "max_tokens")
    sample = (jobs[0].get("input") or {}) if jobs else {}
    if "text" in sample:
        return "characters", "text"
    if "max_tokens" in sample:
        return "tokens", "max_tokens"
    raise ValueError("can't tell what makes a request bigger. The model needs a runtime, or jobs need input.text or input.max_tokens.")


def _size(kind, field, job):
    value = (job.get("input") or {}).get(field)
    if value is None:
        raise ValueError(f"job {job.get('id', '?')} has no input.{field}")
    return len(value) if kind == "characters" else float(value)


def fit(result):
    """Least-squares fit of service_ms = base + slope * size over every measured request.

    Returns a dict with base_ms, per_unit_ms, unit, field, r2, mean_abs_error_pct, samples.
    """
    if result.get("mode") != "live":
        raise ValueError("calibrate needs a live run (a simulation only repeats the formula it was given)")
    workload, model = result["workload"], result["model"]
    by_id = {job["id"]: job for job in workload.get("jobs", [])}
    kind, field = _size_field(model, workload.get("jobs", []))
    xs, ys = [], []
    for item in result.get("results", []):
        for record in item.get("jobs", []):
            job = by_id.get(record["id"])
            if job is None or record.get("service_ms") is None:
                continue
            xs.append(_size(kind, field, job))
            ys.append(float(record["service_ms"]))
    if len(xs) < 3:
        raise ValueError("need at least 3 measured requests to fit")
    n = len(xs)
    mean_x, mean_y = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    if sxx == 0:
        slope = 0.0
    else:
        slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / sxx
    base = mean_y - slope * mean_x
    predicted = [base + slope * x for x in xs]
    ss_res = sum((y - p) ** 2 for y, p in zip(ys, predicted))
    ss_tot = sum((y - mean_y) ** 2 for y in ys)
    r2 = 1.0 - ss_res / ss_tot if ss_tot else 1.0
    errors = [abs(p - y) / y for y, p in zip(ys, predicted) if y > 0]
    return {
        "base_ms": base,
        "per_unit_ms": slope,
        "unit": kind,
        "field": field,
        "r2": r2,
        "mean_abs_error_pct": 100 * sum(errors) / len(errors) if errors else 0.0,
        "samples": n,
    }


def calibrated_model(result, coefficients, name=None):
    """A copy of the run's model with the fitted runtime. Its backend is kept, so it still runs live."""
    model = copy.deepcopy(result["model"])
    base = round(max(0.0, coefficients["base_ms"]), 3)
    slope = round(max(0.0, coefficients["per_unit_ms"]), 4)
    if coefficients["unit"] == "characters":
        model["runtime"] = {"base_ms": base, "per_character_ms": slope, "text_field": coefficients["field"]}
    else:
        model["runtime"] = {"base_ms": base, "per_token_ms": slope, "token_field": coefficients["field"]}
    gpus = ", ".join(gpu["name"] for gpu in result.get("gpus") or []) or result.get("machine", "")
    model["name"] = name or f"{result['model'].get('name', 'model')}-calibrated"
    model["description"] = (
        f"{result['model'].get('name', 'model')} calibrated from a live run on {gpus or 'unknown hardware'} "
        f"({result.get('created', '')}): {coefficients['samples']} requests, R^2 {coefficients['r2']:.3f}."
    )
    model["calibrated_from"] = {
        "created": result.get("created"),
        "target": result.get("target"),
        "gpus": [gpu["name"] for gpu in result.get("gpus") or []],
        "r2": coefficients["r2"],
        "samples": coefficients["samples"],
    }
    return model


def check(result, model):
    """Simulate the run's workload with model and compare mean latency per scheduler.

    Returns rows of (scheduler, live_mean_ms, simulated_mean_ms, error_pct). Schedulers that
    can't be found by name on this machine are skipped.
    """
    from gpu_bench.catalog import resolve
    from gpu_bench.compare import load_select, run_schedule, summarize

    rows = []
    jobs = result["workload"]["jobs"]
    for item in result.get("results", []):
        try:
            path = resolve(item["name"], "scheduler")["path"]
        except ValueError:
            continue
        simulated = summarize(run_schedule(jobs, load_select(path), model, item["name"]))
        live = item["summary"]["mean_latency_ms"]
        error = 100 * (simulated["mean_latency_ms"] - live) / live if live else 0.0
        rows.append((item["name"], live, simulated["mean_latency_ms"], error))
    return rows


def format_report(coefficients, rows):
    unit = "character" if coefficients["unit"] == "characters" else "token"
    lines = [
        f"fit over {coefficients['samples']} requests: "
        f"{coefficients['base_ms']:.1f} ms + {coefficients['per_unit_ms']:.3f} ms per {unit}",
        f"R^2 {coefficients['r2']:.3f}, typical error per request {coefficients['mean_abs_error_pct']:.1f}%",
    ]
    if rows:
        width = max([12, *(len(name) for name, *_rest in rows)])
        lines += ["", f"{'scheduler':<{width}} {'live_ms':>10} {'simulated':>10} {'error':>7}"]
        for name, live, simulated, error in rows:
            lines.append(f"{name:<{width}} {live:10.1f} {simulated:10.1f} {error:+6.1f}%")
    return "\n".join(lines)
