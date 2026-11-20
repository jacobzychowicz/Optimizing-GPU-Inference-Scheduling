# GPU Inference Scheduler

An app for comparing scheduling policies on a GPU model you choose.

You pick a workload, a model, and one or more schedulers. The app runs that same workload through each scheduler and shows which policy fits the model best. You can start from the files in `defaults/`, or replace any of them with your own.

This repository is a skeleton. It has the sample inputs and the scheduler functions. The comparison runner is the next piece to build.

Scheduling here is non-preemptive and request-level: when the GPU is free, the scheduler chooses which waiting request runs next. Dynamic scheduling, where the policy itself can change during a run, can be added later. `select(ready, now_ms)` is already called on every decision, so a later dynamic policy can use the clock and the waiting requests without a new plugin shape.

## Layout

```
defaults/
  workloads/    requests arriving over time
  models/       how long a model takes to run a request
  schedulers/   policies you can compare
gpu_bench/      where the app code will live
```

## Default files

These are ready to copy and edit.

| File | What it is for |
| --- | --- |
| `defaults/workloads/mixed-lengths.json` | A long text-to-speech request, then shorter ones waiting behind it. Pairs with `example-tts`. |
| `defaults/workloads/priority-mix.json` | Same-length requests at different priorities, arriving together. Pairs with `example-tts`. |
| `defaults/workloads/token-mix.json` | A long language-model request, then shorter generations. Pairs with `example-llm`. |
| `defaults/models/example-tts.json` | Stand-in text-to-speech model. Runtime grows with the character count. |
| `defaults/models/example-llm.json` | Stand-in language model. Runtime grows with `max_tokens`. |
| `defaults/schedulers/fifo.py` | First come, first served. |
| `defaults/schedulers/shortest_predicted.py` | Shortest predicted job first. |
| `defaults/schedulers/priority.py` | Higher `priority` first. |
| `defaults/schedulers/aging.py` | Priority that rises the longer a request waits. |

`example-tts` runtime is `40 + 3 * character_count` milliseconds. `example-llm` runtime is `30 + 25 * max_tokens` milliseconds. Each sample job's `predicted_duration_ms` matches that formula, so the shortest-job scheduler and the model agree until you change one of them.

## Workload file

```json
{
  "name": "mixed-lengths",
  "description": "Why this workload exists.",
  "model": "example-tts",
  "jobs": [
    {
      "id": "short-1",
      "arrival_ms": 0,
      "priority": 1,
      "predicted_duration_ms": 58,
      "input": { "text": "Hello." }
    }
  ]
}
```

- `arrival_ms` is when the request shows up, relative to the start of the run.
- `priority` is a number. Higher is more important. The sample schedulers treat a missing priority as `0`.
- `predicted_duration_ms` is the estimate a scheduler can see before the job runs. It can differ from the model's real runtime.
- `input` is what the model reads. Text-to-speech jobs use `text`. Language-model jobs use `prompt` and `max_tokens`.

## Model file

A model file describes a GPU model well enough to compare schedulers before you plug in the real one.

```json
{
  "name": "example-tts",
  "description": "Stand-in text-to-speech model.",
  "runtime": {
    "base_ms": 40,
    "per_character_ms": 3,
    "text_field": "text"
  }
}
```

To try your own model, copy one of these files and change the runtime numbers, or replace the profile later with a module that runs inference on your GPU. The workload format stays the same.

## Scheduler file

Each scheduler is a Python file with one function:

```python
def select(ready, now_ms):
    """Return the next job to run.

    ready: list of job dicts waiting on the GPU
    now_ms: current time in the run
    """
    return ready[0]
```

Copy a file from `defaults/schedulers/`, change `select`, and point a run at your file. Leave the list you were given unchanged.

## What a finished run will answer

Once the runner exists, one workload plus one model plus several schedulers should report, for each policy:

- mean and p95 latency (arrival to finish)
- mean time spent waiting
- throughput

The better policy is whichever one you care about for that model. Shortest-job often wins on average latency and loses on the long requests. Strict priority keeps high-priority work fast and can leave low-priority work waiting. Aging is the sample that pushes back on that.

## Later

- A runner that loads these files and prints the comparison.
- A way to swap the JSON model profile for your own GPU model.
- Dynamic scheduling that can change policy mid-run.
