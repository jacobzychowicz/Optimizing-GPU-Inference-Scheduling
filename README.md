# GPU Inference Scheduler

Find out which scheduling policy serves your GPU model best — on a simulation, on the GPU in front of you, or on a GPU server over SSH.

You pick a workload (requests arriving over time), a model, and one or more schedulers. The app runs that same workload through each scheduler and reports latency, wait, throughput, and GPU use for each, with a timeline of what ran when.

- **Simulate** — instant. The schedule is computed from the model's runtime formula. No GPU needed.
- **Live on GPU** — the workload is replayed on a real clock and each request runs on the model's backend: your own Python code, your own inference server process, or the built-in PyTorch load generator. GPU busy % and peak memory are sampled with `nvidia-smi`.
- **Over SSH** — any of the above on a remote machine. The server needs only Python 3.8+; nothing is installed there. Each run sends the app and your files over one SSH connection, streams progress back, and cleans up after itself.

Scheduling is non-preemptive and request-level: when the GPU is free, the scheduler chooses which waiting request runs next.

## Quick start

```
python -m gpu_bench gui                      # desktop window (pip install PySide6)
python -m gpu_bench                          # terminal menu
python -m gpu_bench mixed-lengths fifo priority
```

Real GPU, this machine (needs PyTorch with CUDA or Apple MPS):

```
python -m gpu_bench gpus                                     # what GPUs and PyTorch are there
python -m gpu_bench token-mix --model synthetic-gpu --live --repeat 3
```

Real GPU, a server:

```
python -m gpu_bench hosts add gpu-box alice@10.0.0.5 --python /opt/venv/bin/python
python -m gpu_bench hosts test gpu-box
python -m gpu_bench token-mix --model synthetic-gpu --live --host gpu-box --save
```

On Linux and macOS, `chmod +x gpu-bench` once and `./gpu-bench` is the same command.

## Desktop window

```
pip install PySide6
python -m gpu_bench gui
```

- **Run on** — *This computer* or any saved SSH host. *Manage hosts...* adds, edits, and tests hosts. *Check GPUs* shows the GPUs, driver, and PyTorch on the chosen machine.
- **Mode** — *Simulate* or *Live on GPU*, and how many repeats to average for live runs.
- Pick a workload (its usual model is selected for you), a model, and schedulers. *Add your own file...* adds a workload, model, or scheduler from anywhere on disk; it stays in the lists next time.
- **Run** works in the background, with a progress bar and the timeline filling in as requests finish. **Stop** ends a live run after the current request.
- Tabs: **Results** (table, winner per metric, timeline — hover a bar for details, *Export...* to CSV/JSON), **Jobs** (every request for one scheduler, sortable), **Log**, and **History** (every finished run, saved automatically; open, export, or delete).

## Terminal menu

`python -m gpu_bench` in a terminal opens a menu:

```
1  model
2  workload
3  schedulers
4  run
5  legacy command line
6  switch simulate / live
7  machine (this one or an SSH host)
q  quit
```

If the shell cannot ask questions (a pipe, a non-interactive SSH command), `python -m gpu_bench` compares every sample instead.

## Commands

```
python -m gpu_bench list                        workloads, models, schedulers
python -m gpu_bench <workload> [scheduler ...]  compare (same as `run`)
    -m/--model NAME     use this model instead of the workload's own
    -s/--scheduler A,B  only these schedulers
    --live              live run on the model's backend (default: simulate)
    --repeat N          live: run each scheduler N times and average
    --host NAME         run on a saved host, or user@host[:port]
    --save              keep the result in the history
    --out FILE          also write .json, or .csv (one row per scheduler)
    --json              print the full result as JSON
python -m gpu_bench gpus [--host NAME]          GPUs, Python, PyTorch
python -m gpu_bench hosts [list|add|remove|test]
python -m gpu_bench history [list|show N|export N FILE]
python -m gpu_bench gui
```

A path works anywhere a name works. `python -m gpu_bench run -h` lists every flag.

## SSH hosts

Login uses your normal `ssh` command, so keys, `ssh-agent`, and `~/.ssh/config` (aliases, `ProxyJump`, ...) all work. Password prompts can't be answered by the app; if you log in with a password, run `ssh-copy-id user@host` once. The first time you connect to a new server, run `ssh user@host` in a terminal once to accept its host key.

A host has:

| Field | Meaning |
| --- | --- |
| host, user, port | Where to connect. host can be an alias from `~/.ssh/config`. |
| key | Private key file, if not your default. |
| python | The Python to run on the server. Used as written, so `/opt/venv/bin/python` or `conda run -n infer python` work. |
| scratch folder | Where a run's files go on the server (default `~/.gpu-bench`). Each run uses its own subfolder and deletes it at the end. |
| ssh options | Extra `-o Key=Value` options. |

Windows needs the OpenSSH client (Settings > Optional features). The server should be Linux or macOS.

## Models and backends

A model file says how long a request takes (for simulation) and, optionally, how to really run it (for live runs).

```json
{
  "name": "my-model",
  "description": "What it is.",
  "runtime": { "base_ms": 30, "per_token_ms": 25, "token_field": "max_tokens" },
  "backend": { "type": "python", "module": "my_backend.py", "config": { "device": "cuda" } }
}
```

`runtime` uses `per_token_ms` (with `token_field`) or `per_character_ms` (with `text_field`). Without a `backend`, a live run sleeps for the runtime formula on a real clock — useful to check a setup or a host before wiring in a model. Without a `runtime`, the model can only run live.

**Python backend** — a file with:

```python
def load(config):          # once: load weights, move to GPU. Returns any state.
    ...
def warmup(state):         # optional
    ...
def infer(state, job):     # once per request. Block until the GPU work is done
    ...                    # (torch.cuda.synchronize()), or latency will be wrong.
def close(state):          # optional
    ...
```

**Worker backend** — a long-running process, in any language, for when your model already runs as its own server:

```json
"backend": { "type": "worker", "command": ["python", "serve.py", "--gpu", "0"], "cwd": "/srv/model" }
```

It prints `{"ready": true}` once loaded, then reads one line `{"job": {...}}` per request and answers with `{"ok": true}` or `{"error": "..."}`. Logs go to stderr. See `defaults/backends/worker_example.py`.

Relative paths are relative to the model file. For SSH runs, a Python backend module and any worker script given by a relative path are sent along automatically; use `cwd` with an absolute path for code that already lives on the server.

## Your own files

Put your workloads, models, and schedulers in `~/.gpu_bench/workloads`, `~/.gpu_bench/models`, and `~/.gpu_bench/schedulers` and they show up by name next to the samples (or pass a path, or use *Add your own file...* in the window). Saved hosts, run history, and window settings also live in `~/.gpu_bench`; set `GPU_BENCH_HOME` to use another folder.

## Sample files

| File | What it is for |
| --- | --- |
| `defaults/workloads/mixed-lengths.json` | A long text-to-speech request, then shorter ones waiting behind it. Pairs with `example-tts`. |
| `defaults/workloads/priority-mix.json` | Same-length requests at different priorities, arriving together. Pairs with `example-tts`. |
| `defaults/workloads/token-mix.json` | A long language-model request, then shorter generations. Pairs with `example-llm`. |
| `defaults/models/example-tts.json` | Stand-in text-to-speech model. Runtime `40 + 3 * characters` ms. |
| `defaults/models/example-llm.json` | Stand-in language model. Runtime `30 + 25 * max_tokens` ms. |
| `defaults/models/synthetic-gpu.json` | Real GPU load: PyTorch matrix multiplies sized by the request. Live only in practice. |
| `defaults/models/example-worker.json` | Same timing as `example-llm`, served by a worker process. A template for your own server. |
| `defaults/backends/torch_matmul.py` | The Python backend behind `synthetic-gpu`. Copy it to start your own. |
| `defaults/backends/worker_example.py` | A minimal worker. |
| `defaults/schedulers/fifo.py` | First come, first served. |
| `defaults/schedulers/shortest_predicted.py` | Shortest predicted job first. |
| `defaults/schedulers/priority.py` | Higher `priority` first. |
| `defaults/schedulers/aging.py` | Priority that rises the longer a request waits. |

## Workload file

```json
{
  "name": "mixed-lengths",
  "description": "Why this workload exists.",
  "model": "example-tts",
  "jobs": [
    { "id": "short-1", "arrival_ms": 0, "priority": 1, "predicted_duration_ms": 58, "input": { "text": "Hello." } }
  ]
}
```

- `arrival_ms` is when the request shows up, relative to the start of the run.
- `priority` is a number. Higher is more important. The sample schedulers treat a missing priority as `0`.
- `predicted_duration_ms` is the estimate a scheduler can see before the job runs. It can differ from the real runtime.
- `input` is what the model reads.

## Scheduler file

```python
def select(ready, now_ms):
    """Return the next job to run.

    ready: list of job dicts waiting on the GPU (don't modify it)
    now_ms: current time in the run (real elapsed time in live runs)
    """
    return ready[0]
```

## What a run reports

For each scheduler:

- mean and p95 latency, arrival to finish (`mean_ms`, `p95_ms`)
- mean time spent waiting (`wait_ms`)
- throughput (`jobs/s`)
- live runs: GPU busy % and peak GPU memory (NVIDIA), and with repeats, the spread of mean latency across runs

Shortest-job often wins on average latency and loses on the long requests. Strict priority keeps high-priority work fast and can starve the rest. Aging pushes back on that.

## Tests

```
python -m unittest discover tests
```

## Later

- Batching and multiple concurrent slots per GPU.
- Preemptive and dynamic policies that change mid-run.
