"""Run with: python -m unittest discover tests"""

import io
import json
import os
import sys
import tarfile
import tempfile
import threading
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gpu_bench import remote, runner, store  # noqa: E402
from gpu_bench.compare import SCHEDULERS, WORKLOADS, MODELS, compare  # noqa: E402

FIFO = [("fifo", SCHEDULERS / "fifo.py")]
SJF = [("shortest_predicted", SCHEDULERS / "shortest_predicted.py")]


class TempHome(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self._old = os.environ.get("GPU_BENCH_HOME")
        os.environ["GPU_BENCH_HOME"] = str(self.dir / "home")

    def tearDown(self):
        if self._old is None:
            os.environ.pop("GPU_BENCH_HOME", None)
        else:
            os.environ["GPU_BENCH_HOME"] = self._old
        self.tmp.cleanup()

    def write(self, name, data):
        path = self.dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(data if isinstance(data, str) else json.dumps(data), encoding="utf-8")
        return path

    def tiny(self, model_extra=None):
        workload = self.write("w.json", {
            "name": "tiny", "model": "m",
            "jobs": [
                {"id": "a", "arrival_ms": 0, "predicted_duration_ms": 60, "input": {"max_tokens": 2}},
                {"id": "b", "arrival_ms": 5, "predicted_duration_ms": 20, "input": {"max_tokens": 0}},
            ],
        })
        model = {"name": "m", "runtime": {"base_ms": 20, "per_token_ms": 20, "token_field": "max_tokens"}}
        model.update(model_extra or {})
        return workload, self.write("m.json", model)


class SimulateTests(TempHome):
    def test_matches_original_compare(self):
        workload, model = WORKLOADS / "token-mix.json", MODELS / "example-llm.json"
        result = runner.execute(workload, model, FIFO + SJF)
        old = compare(workload, model, FIFO + SJF)
        for item in result["results"]:
            self.assertIn(f"{item['summary']['mean_latency_ms']:10.1f}", old)
        self.assertEqual(result["mode"], "simulate")

    def test_events_end_with_result(self):
        events = []
        runner.execute(*self.tiny(), FIFO, emit=events.append)
        self.assertEqual(events[0]["event"], "start")
        self.assertEqual(events[-1]["event"], "result")

    def test_backend_only_model_cannot_simulate(self):
        workload, _ = self.tiny()
        model = self.write("live.json", {"name": "live", "backend": {"type": "worker", "command": ["x"]}})
        with self.assertRaisesRegex(ValueError, "live"):
            runner.execute(workload, model, FIFO)


class LiveTests(TempHome):
    def test_sleep_backend_real_clock(self):
        events = []
        result = runner.execute(*self.tiny(), FIFO, mode="live", emit=events.append)
        jobs = result["results"][0]["jobs"]
        self.assertEqual([job["id"] for job in jobs], ["a", "b"])
        self.assertGreaterEqual(jobs[0]["service_ms"], 55)
        self.assertGreaterEqual(jobs[1]["start_ms"], jobs[0]["finish_ms"])
        self.assertIn("job_start", [event["event"] for event in events])

    def test_python_backend_and_repeats(self):
        self.write("b.py", "calls = []\ndef load(c): return c\ndef infer(s, job): calls.append(job['id'])\n")
        workload, model = self.tiny({"backend": {"type": "python", "module": "b.py", "config": {}}})
        result = runner.execute(workload, model, FIFO + SJF, mode="live", repeats=2)
        self.assertEqual(len(result["results"][0]["runs"]), 2)
        self.assertIn("mean_latency_spread_ms", result["results"][0]["summary"])

    def test_worker_backend(self):
        workload, model = self.tiny({"backend": {"type": "worker", "command": ["python", str(ROOT / "defaults/backends/worker_example.py")]}})
        result = runner.execute(workload, model, FIFO, mode="live")
        self.assertEqual(len(result["results"][0]["jobs"]), 2)

    def test_worker_error_is_reported(self):
        self.write("bad.py", "import json,sys\nprint(json.dumps({'ready': True}), flush=True)\n"
                             "for l in sys.stdin: print(json.dumps({'error': 'boom'}), flush=True)\n")
        workload, model = self.tiny({"backend": {"type": "worker", "command": ["python", "bad.py"]}})
        with self.assertRaisesRegex(RuntimeError, "boom"):
            runner.execute(workload, model, FIFO, mode="live")

    def test_cancel(self):
        cancel = threading.Event()
        cancel.set()
        with self.assertRaises(runner.Cancelled):
            runner.execute(*self.tiny(), FIFO, mode="live", cancel=cancel)


class StoreTests(TempHome):
    def test_rejects_option_injection(self):
        for bad in ({"host": "-oProxyCommand=calc"}, {"host": "ok", "user": "-x"},
                    {"host": "a b"}, {"host": "ok", "ssh_options": ["-F/etc"]}):
            with self.assertRaises(ValueError):
                store.normalize_host(bad)

    def test_save_find_remove(self):
        store.save_host({"name": "box", "host": "10.0.0.5", "user": "me", "port": "2222", "workdir": "~/x"})
        host = store.find_host("box")
        self.assertEqual((host["port"], host["workdir"]), (2222, "x"))
        self.assertEqual(store.find_host("me@h.example:22")["port"], 22)
        store.remove_host("box")
        self.assertEqual(store.load_hosts(), [])

    def test_history_and_export(self):
        result = runner.execute(*self.tiny(), FIFO + SJF)
        store.save_run(result)
        rows = store.list_runs()
        self.assertEqual(rows[0]["workload"], "tiny")
        summary = store.export(result, self.dir / "s.csv")
        jobs = store.export(result, self.dir / "j.csv", "jobs")
        self.assertEqual(len(summary.read_text().strip().splitlines()), 3)
        self.assertEqual(len(jobs.read_text().strip().splitlines()), 5)

    def test_catalog_includes_user_files(self):
        from gpu_bench.catalog import resolve

        self.write("home/workloads/mine.json", {"name": "mine", "jobs": [{"id": "x", "arrival_ms": 0, "input": {}}]})
        self.assertEqual(resolve("mine")["kind"], "workload")


class RemoteTests(TempHome):
    def test_bundle_ships_backend_files(self):
        self.write("b.py", "def infer(s, j): pass\n")
        workload, model = self.tiny({"backend": {"type": "python", "module": "b.py"}})
        data, args = remote.build_bundle(workload, model, FIFO)
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
            names = set(tar.getnames())
            shipped = json.loads(tar.extractfile("job/model.json").read())
        self.assertTrue({"gpu_bench/cli.py", "job/workload.json", "job/backends/b.py", "job/schedulers/fifo.py"} <= names)
        self.assertEqual(shipped["backend"]["module"], "backends/b.py")
        self.assertEqual(args, ["-w", "job/workload.json", "-m", "job/model.json", "-s", "job/schedulers/fifo.py"])

    def test_ssh_command_shape(self):
        host = store.normalize_host({"host": "h.example", "user": "me", "port": 2200})
        os.environ["GPU_BENCH_SSH"] = "ssh"
        try:
            args = remote.ssh_command(host, "true")
        finally:
            os.environ.pop("GPU_BENCH_SSH")
        self.assertEqual(args[-3:], ["--", "me@h.example", "true"])
        self.assertIn("BatchMode=yes", args)
        command = remote.remote_command(host, ["run", "-w", "job/workload.json"])
        self.assertTrue(command.startswith("python3 -c "))

    def test_failure_hints(self):
        host = store.normalize_host({"host": "h"})
        self.assertIn("host key", remote.explain_failure(host, 255, ["Host key verification failed."]))
        self.assertIn("Python was not found", remote.explain_failure(host, 127, ["python3: command not found"]))


if __name__ == "__main__":
    unittest.main()
