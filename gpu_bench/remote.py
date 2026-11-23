"""Run a comparison on another machine over SSH.

Nothing has to be installed on the server except Python 3. Each run sends this
package and the chosen workload, model, and scheduler files to the server in one
SSH connection, runs there, streams progress back as JSON lines, and removes the
files it sent when the run ends.

A host logs in one of two ways:
  key       your `ssh` command, so keys, an agent, and ~/.ssh/config all work.
  password  a password typed when connecting or saved in the system credential
            store. This goes through paramiko instead of the ssh command, and
            new servers are confirmed by fingerprint (trust on first use).
"""

import base64
import hashlib
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
from gpu_bench.store import describe_host, forget_password, get_saved_password, known_hosts_path, set_saved_password


class UnknownHostKey(RuntimeError):
    """A password host whose server key has not been seen before. Confirm, then trust_host_key()."""

    def __init__(self, host, lookup_name, key):
        self.host = host
        self.lookup_name = lookup_name
        self.key = key
        self.key_type = key.get_name()
        digest = hashlib.sha256(key.asbytes()).digest()
        self.fingerprint = "SHA256:" + base64.b64encode(digest).decode("ascii").rstrip("=")
        super().__init__(
            f"{describe_host(host)} is a server this app has not connected to before.\n"
            f"Its {self.key_type} key fingerprint is {self.fingerprint}.\n"
            "Check it with the server's admin (ssh-keygen -lf /etc/ssh/ssh_host_ed25519_key.pub) before trusting it."
        )


class AuthFailed(RuntimeError):
    """The server rejected the user name or password."""


def uses_password(host):
    return host.get("auth") == "password"


def with_password(host, ask):
    """Return a copy of host carrying its password, asking for it when needed.

    ask(host) returns (password, remember) or None to cancel. Hosts that log in
    with keys come back unchanged.
    """
    from gpu_bench.runner import Cancelled

    if not uses_password(host) or host.get("password") is not None:
        return host
    saved = get_saved_password(host["name"]) if host.get("save_password") else None
    if saved:
        return dict(host, password=saved, password_source="saved")
    answer = ask(host)
    if answer is None:
        raise Cancelled("stopped")
    password, remember = answer
    if remember:
        remember_password(host, password)
    return dict(host, password=password, password_source="typed")


def remember_password(host, password):
    """Save the password and mark a saved host to use it from now on."""
    from gpu_bench.store import load_hosts, save_host

    for saved in load_hosts():
        if saved["name"] == host["name"] and not saved["save_password"]:
            save_host(dict(saved, save_password=True))
    set_saved_password(host["name"], password)


def trust_host_key(exc):
    """Remember the server key from an UnknownHostKey so later connections accept it."""
    paramiko = _paramiko()
    path = known_hosts_path()
    keys = paramiko.HostKeys()
    if path.exists():
        keys.load(str(path))
    keys.add(exc.lookup_name, exc.key.get_name(), exc.key)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys.save(str(path))


def _paramiko():
    try:
        import paramiko
    except ImportError as exc:
        raise RuntimeError(
            "password logins need the paramiko package: pip install paramiko. "
            "Or switch this host to key login."
        ) from exc
    import logging

    logger = logging.getLogger("paramiko")
    if not logger.handlers:
        logger.setLevel(logging.CRITICAL)  # errors reach the user as exceptions instead
    return paramiko


def _connect(host):
    import getpass
    import socket

    paramiko = _paramiko()
    if host.get("password") is None:
        raise RuntimeError(f"{describe_host(host)} logs in with a password, and none was given")
    client = paramiko.SSHClient()
    try:
        client.load_system_host_keys()
    except (OSError, paramiko.SSHException):
        pass
    if known_hosts_path().exists():
        try:
            client.load_host_keys(str(known_hosts_path()))
        except (OSError, paramiko.SSHException):
            pass

    class AskFirst(paramiko.MissingHostKeyPolicy):
        def missing_host_key(self, _client, hostname, key):
            raise UnknownHostKey(host, hostname, key)

    client.set_missing_host_key_policy(AskFirst())
    dest = describe_host(host)
    try:
        client.connect(
            hostname=host["host"],
            port=int(host.get("port") or 22),
            username=host.get("user") or getpass.getuser(),
            password=host["password"],
            allow_agent=False,
            look_for_keys=False,
            timeout=15,
            banner_timeout=20,
            auth_timeout=30,
        )
    except UnknownHostKey:
        client.close()
        raise
    except paramiko.BadHostKeyException as exc:
        client.close()
        raise RuntimeError(
            f"WARNING: the key of {dest} has CHANGED since you trusted it. Someone could be intercepting the "
            "connection, or the server was reinstalled. If you know why it changed, remove its line from "
            f"{known_hosts_path()} (or ~/.ssh/known_hosts) and connect again."
        ) from exc
    except paramiko.AuthenticationException as exc:
        client.close()
        message = f"{dest} rejected the user name or password."
        if host.get("password_source") == "saved":
            forget_password(host["name"])
            message += " The saved password was removed, so you will be asked for it next time."
        raise AuthFailed(message) from exc
    except (paramiko.SSHException, OSError, socket.timeout, EOFError) as exc:
        client.close()
        raise RuntimeError(f"cannot connect to {dest}: {exc}") from exc
    return client

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
    if uses_password(host):
        return _stream_password(host, gpu_bench_args, bundle, on_line, cancel)
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


def _stream_password(host, gpu_bench_args, bundle, on_line, cancel=None):
    client = _connect(host)
    stderr = deque(maxlen=60)
    try:
        channel = client.get_transport().open_session()
        channel.exec_command(remote_command(host, gpu_bench_args))
        out = channel.makefile("rb")
        err = channel.makefile_stderr("rb")

        def feed():
            try:
                channel.sendall(bundle)
                channel.shutdown_write()
            except (OSError, EOFError):
                pass

        def drain():
            try:
                for raw in err:
                    stderr.append(raw.decode("utf-8", "replace").rstrip())
            except (OSError, EOFError):
                pass

        def watch():
            while not channel.exit_status_ready() and not channel.closed:
                if cancel is not None and cancel.wait(0.2):
                    client.close()
                    return

        threads = [threading.Thread(target=fn, daemon=True) for fn in (feed, drain, watch)]
        for thread in threads:
            thread.start()
        try:
            for raw in out:
                on_line(raw.decode("utf-8", "replace").rstrip("\r\n"))
        except (OSError, EOFError):
            pass
        code = channel.recv_exit_status() if channel.exit_status_ready() or not channel.closed else -1
        for thread in threads:
            thread.join(timeout=2)
        return code, list(stderr)
    finally:
        client.close()


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
