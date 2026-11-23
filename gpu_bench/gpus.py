"""Find the GPUs on this machine and sample their use while a live run is going."""

import importlib.util
import platform
import shutil
import socket
import subprocess
import sys
import threading


def quiet_popen_kwargs():
    """Keep Windows from flashing a console window when the desktop app starts a tool."""
    if sys.platform == "win32":
        return {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0)}
    return {}


def _number(text):
    try:
        value = float(text)
    except (TypeError, ValueError):
        return None
    return int(value) if value.is_integer() else value


def query_nvidia():
    """One entry per NVIDIA GPU, from nvidia-smi. Empty when there is none."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return []
    fields = "index,name,memory.total,memory.used,utilization.gpu,driver_version"
    try:
        out = subprocess.run(
            [exe, f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=15,
            **quiet_popen_kwargs(),
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if out.returncode != 0:
        return []
    gpus = []
    for line in out.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 6:
            continue
        gpus.append(
            {
                "index": _number(parts[0]),
                "name": parts[1],
                "vendor": "nvidia",
                "memory_total_mb": _number(parts[2]),
                "memory_used_mb": _number(parts[3]),
                "utilization_pct": _number(parts[4]),
                "driver": parts[5],
            }
        )
    return gpus


def torch_info():
    """Whether PyTorch is installed and which accelerators it can see."""
    if importlib.util.find_spec("torch") is None:
        return {"installed": False}
    try:
        import torch
    except Exception as exc:  # a broken install should not stop the probe
        return {"installed": True, "error": str(exc)}
    info = {
        "installed": True,
        "version": torch.__version__,
        "cuda": bool(torch.cuda.is_available()),
        "cuda_version": getattr(torch.version, "cuda", None),
        "devices": [],
        "mps": False,
    }
    if info["cuda"]:
        info["devices"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    mps = getattr(torch.backends, "mps", None)
    if mps is not None:
        try:
            info["mps"] = bool(mps.is_available())
        except Exception:
            info["mps"] = False
    return info


def probe(check_torch=True):
    return {
        "machine": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "gpus": query_nvidia(),
        "torch": torch_info() if check_torch else None,
    }


def format_probe(info):
    lines = [
        f"machine: {info.get('machine', '?')}",
        f"platform: {info.get('platform', '?')}",
        f"python: {info.get('python', '?')}",
    ]
    gpus = info.get("gpus") or []
    if gpus:
        lines.append("gpus:")
        for gpu in gpus:
            lines.append(
                f"  [{gpu['index']}] {gpu['name']}  "
                f"{gpu['memory_used_mb']}/{gpu['memory_total_mb']} MB used  "
                f"{gpu['utilization_pct']}% busy  driver {gpu['driver']}"
            )
    else:
        lines.append("gpus: no NVIDIA GPU found (nvidia-smi is missing or reports none)")
    torch = info.get("torch")
    if torch is not None:
        if not torch.get("installed"):
            lines.append("torch: not installed (needed only for the synthetic-gpu model)")
        elif torch.get("error"):
            lines.append(f"torch: installed but failed to import: {torch['error']}")
        else:
            seen = []
            if torch.get("cuda"):
                seen.append(f"CUDA {torch.get('cuda_version')}: {', '.join(torch.get('devices') or [])}")
            if torch.get("mps"):
                seen.append("Apple MPS")
            lines.append(f"torch: {torch.get('version')}  " + ("; ".join(seen) if seen else "CPU only"))
    return "\n".join(lines)


def short_gpu_label(info):
    """One line for a status bar."""
    gpus = info.get("gpus") or []
    if gpus:
        names = {}
        for gpu in gpus:
            names[gpu["name"]] = names.get(gpu["name"], 0) + 1
        return ", ".join(f"{count}x {name}" if count > 1 else name for name, count in names.items())
    torch = info.get("torch") or {}
    if torch.get("mps"):
        return "Apple GPU (MPS)"
    return "no GPU detected"


class Sampler:
    """Reads nvidia-smi in the background. stop() returns mean busy % and peak memory."""

    def __init__(self, interval_ms=250):
        self.interval_ms = interval_ms
        self._proc = None
        self._thread = None
        self._util = []
        self._mem_by_tick = {}

    def start(self):
        exe = shutil.which("nvidia-smi")
        if not exe:
            return self
        try:
            self._proc = subprocess.Popen(
                [exe, "--query-gpu=timestamp,index,utilization.gpu,memory.used",
                 "--format=csv,noheader,nounits", f"-lms={self.interval_ms}"],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                **quiet_popen_kwargs(),
            )
        except OSError:
            self._proc = None
            return self
        self._thread = threading.Thread(target=self._read, daemon=True)
        self._thread.start()
        return self

    def _read(self):
        for line in self._proc.stdout:
            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 4:
                continue
            util, mem = _number(parts[2]), _number(parts[3])
            if util is not None:
                self._util.append(util)
            if mem is not None:
                self._mem_by_tick[parts[0]] = self._mem_by_tick.get(parts[0], 0) + mem

    def stop(self):
        if self._proc is None:
            return {}
        self._proc.terminate()
        try:
            self._proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            self._proc.kill()
        if self._thread:
            self._thread.join(timeout=3)
        stats = {}
        if self._util:
            stats["gpu_util_pct"] = sum(self._util) / len(self._util)
        if self._mem_by_tick:
            stats["gpu_mem_peak_mb"] = max(self._mem_by_tick.values())
        return stats
