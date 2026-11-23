# GPU Inference Scheduler

Compare scheduling policies for a GPU inference model: run the same workload through each scheduler and see which gives the best latency, wait, throughput, and GPU use.

- **Simulate**: instant, computed from the model's runtime formula. No GPU needed.
- **Live on GPU**: replays the workload in real time on a real backend (your Python code, your inference server, or a built-in PyTorch load generator) and samples GPU use.
- **Over SSH**: either mode on a remote server. The server needs only Python 3.8+.

Scheduling is non-preemptive and request-level: whenever the GPU is free, the scheduler picks which waiting request runs next.

## Setup

### 1. Get the app

| You want | Do this |
| --- | --- |
| Just the desktop app (Windows) | Download `GPU-Inference-Scheduler.exe` from this repo and double-click it. Nothing else to install. |
| Everything (any OS) | `git clone https://github.com/jacobzychowicz/Optimizing-GPU-Inference-Scheduling.git` with Python 3.8+ installed. |

### 2. Install what you need (Python install only)

The command line works with no packages at all. Add only what you use:

| Package | Needed for | Install |
| --- | --- | --- |
| PySide6 | the desktop window | `pip install -r requirements-gui.txt` (installs all three) |
| paramiko, keyring | SSH hosts that log in with a password | |
| PyTorch | live runs of the `synthetic-gpu` model on this machine | [pytorch.org](https://pytorch.org/get-started/locally/) (choose a CUDA build) |

### 3. Check it works

```
python -m gpu_bench mixed-lengths fifo priority    # simulation: prints a comparison table
python -m gpu_bench gpus                           # lists GPUs, driver, and PyTorch/CUDA
python -m gpu_bench gui                            # opens the window
```

On Linux/macOS you can also `chmod +x gpu-bench` and use `./gpu-bench` in place of `python -m gpu_bench`.

### 4. Run on a GPU

**This machine** (NVIDIA + CUDA PyTorch, or Apple MPS):

```
python -m gpu_bench token-mix --model synthetic-gpu --live --repeat 3
```

**A remote server**: needs SSH access and Python 3.8+ on a Linux/macOS server. Nothing is installed there: each run copies the app over, runs it, streams results back, and cleans up.

```
# Key login (uses your ssh keys, agent, and ~/.ssh/config). Connect once with `ssh` first to accept the host key.
python -m gpu_bench hosts add gpu-box alice@10.0.0.5 --python /opt/venv/bin/python

# Password login: asked each time, or --save-password to keep it in Windows Credential Manager / Keychain
python -m gpu_bench hosts add lab bob@lab.example.edu --password --save-password

python -m gpu_bench hosts test gpu-box                                   # connect and show its GPUs
python -m gpu_bench token-mix --model synthetic-gpu --live --host gpu-box --save
```

In the window, the same steps are under **Run on → Manage hosts... → Test connection**.

Notes:
- Key logins on Windows need the OpenSSH client (Settings → Optional features).
- A password login shows a new server's key fingerprint once and asks you to trust it. A saved password the server rejects is deleted, and you're asked again.
- For scripts with no terminal, set `GPU_BENCH_SSH_PASSWORD`.
- The `.exe` has no PyTorch, so `synthetic-gpu` runs live only on SSH hosts. Worker backends need Python installed.

## Using it

**Desktop window**: choose **Run on** (this computer or a host), **Simulate** or **Live on GPU**, then a workload, a model, and schedulers, and press **Run**. Results show as a table, the best scheduler for each metric, and a timeline (hover for details). Tabs: **Jobs** (per request), **Log**, and **History** (every run, saved automatically). Export to CSV/JSON from Results or History. **Add your own file...** adds workloads, models, or schedulers from anywhere on disk.

**Terminal menu**: `python -m gpu_bench` with no arguments, in a terminal.

**Commands**:

```
python -m gpu_bench list                          workloads, models, schedulers
python -m gpu_bench <workload> [scheduler ...]    compare
    -m NAME       model (default: the workload's own)
    -s A,B        only these schedulers
    --live        live run (default: simulate)
    --repeat N    live: average over N runs
    --host NAME   saved host, or user@host[:port]
    --save        keep in history
    --out FILE    write .json, or .csv summary
    --json        print the full result as JSON
python -m gpu_bench gpus [--host NAME]
python -m gpu_bench hosts [list | add | remove | test]
python -m gpu_bench history [list | show N | export N FILE]
```

A file path works anywhere a name works. Add `-h` to any command for every option.

## Your own files

Put them in `~/.gpu_bench/workloads`, `models`, or `schedulers` and they appear by name next to the samples. That folder also holds saved hosts, history, trusted server keys, and settings; set `GPU_BENCH_HOME` to move it. Passwords are never written there.

**Workload**: requests arriving over time.

```json
{ "name": "my-load", "model": "example-llm",
  "jobs": [ { "id": "a", "arrival_ms": 0, "priority": 1, "predicted_duration_ms": 430, "input": { "max_tokens": 16 } } ] }
```

`priority`: higher runs first (missing means 0). `predicted_duration_ms`: what a scheduler sees before a request runs. `input`: what the model reads.

**Scheduler**: one function. Don't modify `ready`.

```python
def select(ready, now_ms):
    return min(ready, key=lambda job: job["arrival_ms"])
```

**Model**: `runtime` is used for simulation; `backend` (optional) runs requests for real in live mode. Without a backend, a live run just waits for the formula's time, which is a quick way to test a setup or host.

```json
{ "name": "my-model",
  "runtime": { "base_ms": 30, "per_token_ms": 25, "token_field": "max_tokens" },
  "backend": { "type": "python", "module": "my_backend.py", "config": { "device": "cuda" } } }
```

`runtime` takes `per_token_ms` + `token_field`, or `per_character_ms` + `text_field`. Backends:

- **python**: a file with `load(config) -> state` and `infer(state, job)`, plus optional `warmup(state)` and `close(state)`. `infer` must block until the GPU work finishes (`torch.cuda.synchronize()`). Start from `defaults/backends/torch_matmul.py`.
- **worker**: `{"type": "worker", "command": ["python", "serve.py"], "cwd": "/srv/model"}`. This is a long-running process in any language. It prints `{"ready": true}` once loaded, then answers each `{"job": {...}}` line with `{"ok": true}` or `{"error": "..."}`. Start from `defaults/backends/worker_example.py`.

Paths are relative to the model file. For SSH runs, relative backend files are copied over automatically; use an absolute `cwd` for code that already lives on the server.

## Samples

| | |
| --- | --- |
| Workloads | `mixed-lengths` (long TTS request blocking short ones), `priority-mix` (equal lengths, mixed priorities), `token-mix` (long LLM generation, then short ones) |
| Models | `example-tts`, `example-llm` (formula only), `synthetic-gpu` (real PyTorch GPU load), `example-worker` (worker template) |
| Schedulers | `fifo`, `shortest_predicted`, `priority`, `aging` (priority that grows with waiting time) |

## What a run reports

Per scheduler: mean and p95 latency (arrival to finish), mean wait, and jobs/s. Live runs add GPU busy % and peak memory (NVIDIA), plus the spread across repeats.

Typical trade-offs: shortest-job wins on average latency but makes long requests wait; strict priority can starve low-priority work; aging balances the two.

## Development

```
python -m unittest discover tests                  # tests
pip install pyinstaller && python build_exe.py     # rebuild GPU-Inference-Scheduler.exe
```

Planned: batching and concurrent slots per GPU, preemptive and dynamic policies.
