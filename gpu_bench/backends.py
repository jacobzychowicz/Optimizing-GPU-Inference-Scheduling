"""What actually runs a request in a live run.

A model file picks its backend with a "backend" block. Without one, the
model's runtime formula is used and each request just sleeps for that long,
which is handy for checking a setup (or an SSH host) before a real model is
wired in.

  "backend": {"type": "python", "module": "my_model.py", "config": {...}}
      my_model.py defines load(config) -> state and infer(state, job).
      Optional: warmup(state), close(state). infer must block until the GPU
      work for the job is finished (call torch.cuda.synchronize(), for example).

  "backend": {"type": "worker", "command": ["python", "worker.py"], "cwd": "..."}
      A long-running process. It prints {"ready": true} once its model is loaded,
      then reads one JSON line per request ({"job": {...}}) and answers with one
      line ({"ok": true} or {"error": "..."}). Any language works.

Relative paths are relative to the model file. "python" as the first word of a
worker command means the Python that runs gpu-bench.
"""

import importlib.util
import json
import shlex
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path

from gpu_bench.compare import duration_ms
from gpu_bench.gpus import quiet_popen_kwargs


def describe(model):
    backend = model.get("backend")
    if not backend:
        return "timing formula (sleeps; no GPU work)"
    kind = backend.get("type")
    if kind == "python":
        return f"python module {backend.get('module')}"
    if kind == "worker":
        return "worker process"
    return f"unknown backend {kind}"


def can_simulate(model):
    return "runtime" in model


def open_backend(model, model_path):
    backend = model.get("backend")
    base = Path(model_path).resolve().parent
    if not backend:
        if "runtime" not in model:
            raise ValueError(f"{model.get('name', 'model')} has neither a runtime formula nor a backend")
        return SleepBackend(model)
    kind = backend.get("type")
    if kind == "python":
        return PythonBackend(model, backend, base)
    if kind == "worker":
        return WorkerBackend(model, backend, base)
    raise ValueError(f"{model.get('name', 'model')} has unknown backend type '{kind}'. Use python or worker.")


def _python_executable():
    """The Python running gpu-bench, or a Python on PATH when running from the packaged .exe."""
    if not getattr(sys, "frozen", False):
        return sys.executable
    import shutil

    for name in ("python", "python3", "py"):
        found = shutil.which(name)
        if found and "WindowsApps" not in found:
            return found
    raise ValueError("this worker needs Python, and none was found on PATH. Install Python 3 or use a full path.")


class SleepBackend:
    """Sleeps for the runtime formula. Same timing as a simulation, but on a real clock."""

    def __init__(self, model):
        self.model = model

    def warmup(self):
        pass

    def run(self, job):
        time.sleep(duration_ms(self.model, job) / 1000.0)

    def close(self):
        pass


class PythonBackend:
    def __init__(self, model, backend, base):
        module_path = backend.get("module")
        if not module_path:
            raise ValueError(f"{model.get('name', 'model')}: a python backend needs a module path")
        path = (base / module_path).resolve()
        if not path.is_file():
            raise ValueError(f"backend module {path} does not exist")
        spec = importlib.util.spec_from_file_location(f"gpu_bench_backend_{path.stem}", path)
        if spec is None or spec.loader is None:
            raise ValueError(f"cannot load backend module {path}")
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception as exc:
            raise ValueError(f"cannot load backend module {path}: {exc}") from exc
        if not callable(getattr(module, "infer", None)):
            raise ValueError(f"{path} must define infer(state, job)")
        self.module = module
        self.path = path
        loader = getattr(module, "load", None)
        try:
            self.state = loader(dict(backend.get("config") or {})) if callable(loader) else None
        except Exception as exc:
            raise RuntimeError(f"{path.name} load() failed: {exc}") from exc

    def warmup(self):
        warm = getattr(self.module, "warmup", None)
        if callable(warm):
            warm(self.state)

    def run(self, job):
        try:
            return self.module.infer(self.state, job)
        except Exception as exc:
            raise RuntimeError(f"{self.path.name} failed on job {job.get('id', '?')}: {exc}") from exc

    def close(self):
        closer = getattr(self.module, "close", None)
        if callable(closer):
            try:
                closer(self.state)
            except Exception:
                pass


class WorkerBackend:
    def __init__(self, model, backend, base, ready_timeout_s=600):
        command = backend.get("command")
        if isinstance(command, str):
            command = shlex.split(command)
        if not command:
            raise ValueError(f"{model.get('name', 'model')}: a worker backend needs a command")
        command = list(command)
        if command[0] in ("python", "python3"):
            command[0] = _python_executable()
        cwd = Path(backend["cwd"]).expanduser() if backend.get("cwd") else base
        if not cwd.is_absolute():
            cwd = (base / cwd).resolve()
        self.name = model.get("name", "worker")
        self._stderr = deque(maxlen=40)
        try:
            self.proc = subprocess.Popen(
                command,
                cwd=str(cwd),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                **quiet_popen_kwargs(),
            )
        except OSError as exc:
            raise ValueError(f"cannot start worker {command[0]}: {exc}") from exc
        threading.Thread(target=self._drain_stderr, daemon=True).start()
        reply = self._read_reply(ready_timeout_s)
        if not reply.get("ready"):
            self.close()
            raise RuntimeError(f"worker for {self.name} did not report ready: {reply}")

    def _drain_stderr(self):
        for line in self.proc.stderr:
            self._stderr.append(line.rstrip())

    def _read_reply(self, timeout_s=None):
        box = {}

        def read():
            box["line"] = self.proc.stdout.readline()

        reader = threading.Thread(target=read, daemon=True)
        reader.start()
        reader.join(timeout_s)
        if reader.is_alive():
            self.proc.kill()
            raise RuntimeError(f"worker for {self.name} did not answer within {timeout_s} s")
        line = box.get("line", "")
        if not line:
            self.proc.wait(timeout=5)
            tail = "\n".join(self._stderr) or "(no output)"
            raise RuntimeError(f"worker for {self.name} exited with code {self.proc.returncode}:\n{tail}")
        try:
            return json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"worker for {self.name} printed something that is not JSON: {line.strip()[:200]}") from exc

    def warmup(self):
        pass

    def run(self, job):
        try:
            self.proc.stdin.write(json.dumps({"job": job}) + "\n")
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise RuntimeError(f"worker for {self.name} is gone: {exc}") from exc
        reply = self._read_reply()
        if reply.get("error"):
            raise RuntimeError(f"worker for {self.name} failed on job {job.get('id', '?')}: {reply['error']}")
        return reply

    def close(self):
        if self.proc.poll() is None:
            try:
                self.proc.stdin.close()
            except OSError:
                pass
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=5)
        for stream in (self.proc.stdin, self.proc.stdout, self.proc.stderr):
            try:
                stream.close()
            except (OSError, ValueError):
                pass
