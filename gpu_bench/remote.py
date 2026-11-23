"""Run a comparison on another machine over SSH.

Nothing has to be installed on the server except Python 3. Each run sends this
package and the chosen workload, model, and scheduler files to the server in one
SSH connection, runs there, streams progress back as JSON lines, and removes the
files it sent when the run ends.

Authentication is whatever your `ssh` command already uses: keys, an agent, or
entries in ~/.ssh/config. Password prompts cannot be answered from here, so set
up a key first (ssh-copy-id user@host).
"""

import base64
import io
import json
import os
import re
import shlex
import shutil
import subprocess
import tarfile
import threading
import time
import uuid
from collections import deque
from pathlib import Path

from gpu_bench.compare import load_json
from gpu_bench.gpus import quiet_popen_kwargs
from gpu_bench.store import describe_host

PACKAGE = Path(__file__).resolve().parent

# Runs on the server. argv: b64-of-this, workdir, run id, gpu-bench arguments...
_BOOTSTRAP = r"""
import io, os, shutil, sys, tarfile
base, run_id, args = os.path.abspath(os.path.expanduser(sys.argv[2])), sys.argv[3], sys.argv[4:]
data = sys.stdin.buffer.read()
run = os.path.join(base, run_id)
os.makedirs(run, exist_ok=True)
code = 1
try:
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        for m in tar.getmembers():
            parts = m.name.split("/")
            if m.name.startswith("/") or ".." in parts or not (m.isfile() or m.isdir()):
                raise SystemExit("gpu-bench: refusing unsafe bundle entry " + m.name)
        kw = {"filter": "data"} if hasattr(tarfile, "data_filter") else {}
        tar.extractall(run, **kw)
    sys.path.insert(0, run)
    os.chdir(run)
    from gpu_bench.cli import main
    code = main(args)
finally:
    os.chdir(base)
    shutil.rmtree(run, ignore_errors=True)
sys.exit(code)
"""


def ssh_executable():
    exe = os.environ.get("GPU_BENCH_SSH") or shutil.which("ssh")
    if not exe:
        raise RuntimeError(
            "the ssh command was not found. Install an OpenSSH client "
            "(Windows: Settings > Optional features > OpenSSH Client)."
        )
    return exe


def ssh_command(host, remote_command):
    args = [ssh_executable(), "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=30"]
    if host.get("port"):
        args += ["-p", str(host["port"])]
    if host.get("key"):
        args += ["-i", str(Path(host["key"]).expanduser())]
    for option in host.get("ssh_options") or []:
        args += ["-o", option]
    dest = f"{host['user']}@{host['host']}" if host.get("user") else host["host"]
    args += ["--", dest, remote_command]
    return args


def remote_command(host, gpu_bench_args):
    """The shell line run on the server. host['python'] is used as written, so it can be
    a path or something like `conda run -n infer python`."""
    code = base64.b64encode(_BOOTSTRAP.encode("utf-8")).decode("ascii")
    run_id = "run-" + uuid.uuid4().hex[:12]
    parts = [host.get("python") or "python3", "-c", shlex.quote("import base64,sys;exec(base64.b64decode(sys.argv[1]))"),
             code, shlex.quote(host.get("workdir") or ".gpu-bench"), run_id]
    parts += [shlex.quote(arg) for arg in gpu_bench_args]
    return " ".join(parts)


# bundle -------------------------------------------------------------------


def _safe(name):
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "_", name).strip(".") or "file"
    return cleaned


def build_bundle(workload_path=None, model_path=None, schedulers=()):
    """A .tar.gz with gpu_bench/ and job/. Returns (bytes, gpu-bench args for the files)."""
    buffer = io.BytesIO()
    args = []
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for path in sorted(PACKAGE.glob("*.py")):
            tar.add(str(path), arcname=f"gpu_bench/{path.name}")

        def add_bytes(name, data):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mtime = int(time.time())
            info.mode = 0o644
            tar.addfile(info, io.BytesIO(data))

        if workload_path is not None:
            add_bytes("job/workload.json", Path(workload_path).read_bytes())
            args += ["-w", "job/workload.json"]
        if model_path is not None:
            model = load_json(model_path)
            model = _bundle_backend_files(model, Path(model_path), add_bytes)
            add_bytes("job/model.json", json.dumps(model, indent=2).encode("utf-8"))
            args += ["-m", "job/model.json"]
        used = set()
        for name, path in schedulers:
            stem = _safe(name)
            while stem in used:
                stem += "_"
            used.add(stem)
            add_bytes(f"job/schedulers/{stem}.py", Path(path).read_bytes())
            args += ["-s", f"job/schedulers/{stem}.py"]
    return buffer.getvalue(), args


def _bundle_backend_files(model, model_path, add_bytes):
    """Send the code a model's backend needs and point the copied model at it."""
    backend = model.get("backend")
    if not backend:
        return model
    model = json.loads(json.dumps(model))
    backend = model["backend"]
    base = model_path.resolve().parent

    def ship(relative):
        source = (base / relative).resolve()
        target = f"backends/{_safe(source.name)}"
        add_bytes(f"job/{target}", source.read_bytes())
        return target

    if backend.get("type") == "python" and backend.get("module"):
        source = (base / backend["module"])
        if not source.is_file():
            raise ValueError(f"backend module {source} does not exist")
        backend["module"] = ship(backend["module"])
    elif backend.get("type") == "worker" and not backend.get("cwd"):
        command = backend.get("command") or []
        if isinstance(command, str):
            command = shlex.split(command)
        rewritten = []
        for word in command:
            candidate = base / word
            if not Path(word).is_absolute() and candidate.is_file():
                rewritten.append(ship(word))
            else:
                rewritten.append(word)
        backend["command"] = rewritten
    return model


# running ------------------------------------------------------------------


def _stream(host, gpu_bench_args, bundle, on_line, cancel=None):
    """Run gpu-bench on the server. Calls on_line for each stdout line. Returns (code, stderr tail)."""
    command = ssh_command(host, remote_command(host, gpu_bench_args))
    try:
        proc = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            **quiet_popen_kwargs(),
        )
    except OSError as exc:
        raise RuntimeError(f"cannot start ssh: {exc}") from exc

    stderr = deque(maxlen=60)

    def feed():
        try:
            proc.stdin.write(bundle)
            proc.stdin.close()
        except (BrokenPipeError, OSError):
            pass

    def drain():
        for raw in proc.stderr:
            stderr.append(raw.decode("utf-8", "replace").rstrip())

    def watch():
        while proc.poll() is None:
            if cancel is not None and cancel.wait(0.2):
                proc.kill()
                return

    threads = [threading.Thread(target=fn, daemon=True) for fn in (feed, drain, watch)]
    for thread in threads:
        thread.start()
    for raw in proc.stdout:
        on_line(raw.decode("utf-8", "replace").rstrip("\r\n"))
    proc.wait()
    for thread in threads:
        thread.join(timeout=2)
    return proc.returncode, list(stderr)


def explain_failure(host, code, stderr):
    text = "\n".join(stderr)
    dest = describe_host(host)
    hints = []
    lowered = text.lower()
    if "host key verification failed" in lowered or "no matching host key" in lowered:
        hints.append(f"Connect once in a terminal (ssh {dest}) to check and accept the server's host key.")
    if "permission denied" in lowered:
        hints.append("The server refused the login. Set up a key (ssh-copy-id) or pick the key file for this host.")
    if "could not resolve hostname" in lowered:
        hints.append("The hostname could not be found. Check the spelling, or use an IP address.")
    if "connection refused" in lowered or "timed out" in lowered:
        hints.append("The server did not accept the connection. Check the address, port, VPN, and firewall.")
    if code == 127 or "command not found" in lowered or "no such file or directory" in lowered and "python" in lowered:
        hints.append(f"Python was not found on {dest}. Set this host's Python (for example python3 or a venv path).")
    if "syntaxerror" in lowered:
        hints.append("The server's Python is too old. gpu-bench needs Python 3.8 or newer.")
    tail = "\n".join(stderr[-15:]) or "(ssh printed nothing)"
    message = f"run on {dest} failed (exit code {code}).\n{tail}"
    if hints:
        message += "\n\n" + "\n".join(hints)
    return message


def run(host, workload_path, model_path, schedulers, mode="simulate", repeats=1, emit=None, cancel=None):
    """Same contract as runner.execute, but the run happens on host."""
    from gpu_bench.runner import Cancelled

    emit = emit or (lambda _event: None)
    bundle, file_args = build_bundle(workload_path, model_path, schedulers)
    args = ["run", "--events", "--mode", mode, "--repeat", str(repeats), *file_args]
    emit({"event": "log", "message": f"Connecting to {describe_host(host)}"})
    box = {}

    def on_line(line):
        if not line.startswith("{"):
            if line.strip():
                emit({"event": "log", "message": line})
            return
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            emit({"event": "log", "message": line})
            return
        if event.get("event") == "result":
            box["result"] = event["data"]
        else:
            emit(event)

    code, stderr = _stream(host, args, bundle, on_line, cancel)
    if cancel is not None and cancel.is_set():
        raise Cancelled("stopped")
    result = box.get("result")
    if result is None or code != 0:
        raise RuntimeError(explain_failure(host, code, stderr))
    result["target"] = host["name"]
    result["host"] = describe_host(host)
    emit({"event": "result", "data": result})
    return result


def probe(host, check_torch=True):
    """GPU and Python details from the server, as gpus.probe() returns them."""
    bundle, _args = build_bundle()
    args = ["gpus", "--json"] + ([] if check_torch else ["--no-torch"])
    lines = []
    code, stderr = _stream(host, args, bundle, lines.append)
    text = "\n".join(lines).strip()
    if code != 0 or not text:
        raise RuntimeError(explain_failure(host, code, stderr))
    try:
        info = json.loads(text[text.index("{"):])
    except (ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"unexpected answer from {describe_host(host)}: {text[:300]}") from exc
    info["host"] = describe_host(host)
    return info
