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

    def test_hosts_add_reads_port_from_target(self):
        from contextlib import redirect_stdout

        from gpu_bench.cli import main

        with redirect_stdout(io.StringIO()):
            self.assertEqual(main(["hosts", "add", "box", "me@10.0.0.5:2222"]), 0)
        host = store.find_host("box")
        self.assertEqual((host["host"], host["port"], host["user"]), ("10.0.0.5", 2222, "me"))

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


class PasswordTests(TempHome):
    def setUp(self):
        super().setUp()
        self.vault = {}
        self._orig = (store.get_saved_password, store.set_saved_password, store.forget_password, store.can_save_passwords)
        store.get_saved_password = lambda name: self.vault.get(name)
        store.set_saved_password = lambda name, pw: self.vault.__setitem__(name, pw)
        store.forget_password = lambda name: self.vault.pop(name, None)
        store.can_save_passwords = lambda: True
        remote.get_saved_password = store.get_saved_password
        remote.set_saved_password = store.set_saved_password

    def tearDown(self):
        (store.get_saved_password, store.set_saved_password, store.forget_password, store.can_save_passwords) = self._orig
        remote.get_saved_password, remote.set_saved_password = self._orig[0], self._orig[1]
        super().tearDown()

    def test_password_never_written_to_hosts_file(self):
        store.save_host({"name": "p", "host": "h", "auth": "password", "save_password": True, "password": "hunter2"})
        text = (store.home() / "hosts.json").read_text()
        self.assertNotIn("hunter2", text)
        self.assertEqual(store.load_hosts()[0]["auth"], "password")

    def test_key_hosts_never_ask(self):
        host = store.normalize_host({"host": "h"})
        self.assertIs(remote.with_password(host, lambda h: self.fail("asked")), host)

    def test_ask_each_time_then_remember(self):
        store.save_host({"name": "p", "host": "h", "auth": "password"})
        host = store.find_host("p")
        typed = remote.with_password(host, lambda h: ("pw1", False))
        self.assertEqual((typed["password"], typed["password_source"]), ("pw1", "typed"))
        self.assertEqual(self.vault, {})
        remote.with_password(host, lambda h: ("pw2", True))
        self.assertEqual(self.vault, {"p": "pw2"})
        host = store.find_host("p")
        self.assertTrue(host["save_password"])
        saved = remote.with_password(host, lambda h: self.fail("asked again"))
        self.assertEqual((saved["password"], saved["password_source"]), ("pw2", "saved"))

    def test_cancel_prompt(self):
        host = store.normalize_host({"host": "h", "auth": "password"})
        with self.assertRaises(runner.Cancelled):
            remote.with_password(host, lambda h: None)

    def test_turning_off_save_and_removing_forget_password(self):
        store.save_host({"name": "p", "host": "h", "auth": "password", "save_password": True})
        self.vault["p"] = "pw"
        store.save_host({"name": "p", "host": "h", "auth": "password", "save_password": False})
        self.assertEqual(self.vault, {})
        store.save_host({"name": "p", "host": "h", "auth": "password", "save_password": True})
        self.vault["p"] = "pw"
        store.save_host({"name": "q", "host": "h", "auth": "password", "save_password": True}, replace="p")
        self.assertEqual(self.vault, {"q": "pw"})
        store.remove_host("q")
        self.assertEqual(self.vault, {})


class NarrationTests(TempHome):
    def setUp(self):
        super().setUp()
        from gpu_bench import workloads

        self.data = workloads.generate_narration("n", users=3, script_users=2, segments=(20, 30), retakes=5, seed=3)
        self.path = self.write("n.json", self.data)

    def test_generator_is_deterministic_and_valid(self):
        from gpu_bench import workloads
        from gpu_bench.compare import check_jobs

        again = workloads.generate_narration("n", users=3, script_users=2, segments=(20, 30), retakes=5, seed=3)
        self.assertEqual(again, self.data)
        check_jobs(self.data, "n")
        arrivals = [job["arrival_ms"] for job in self.data["jobs"]]
        self.assertEqual(arrivals, sorted(arrivals))
        kinds = {job["kind"] for job in self.data["jobs"]}
        self.assertEqual(kinds, {"script_segment", "retake"})
        for user in ("user-1", "user-2"):
            burst = {job["arrival_ms"] for job in self.data["jobs"] if job["user"] == user and job["kind"] == "script_segment"}
            self.assertEqual(len(burst), 1)

    def test_traffic_generator(self):
        from gpu_bench import workloads
        from gpu_bench.compare import check_jobs

        for pattern in workloads.PATTERNS:
            data = workloads.generate_traffic("t", count=50, pattern=pattern, seed=1)
            check_jobs(data, "t")
            self.assertEqual(len(data["jobs"]), 50)

    def test_fair_schedulers_rescue_retakes(self):
        model = MODELS / "example-tts.json"
        schedulers = [(name, SCHEDULERS / f"{name}.py") for name in ("fifo", "fair_share", "retake_first")]
        result = runner.execute(self.path, model, schedulers)
        retake = {item["name"]: item["summary"]["by_kind"]["retake"]["mean_latency_ms"] for item in result["results"]}
        self.assertLess(retake["fair_share"], retake["fifo"] / 3)
        self.assertLess(retake["retake_first"], retake["fifo"] / 3)

    def test_by_kind_matches_overall(self):
        result = runner.execute(self.path, MODELS / "example-tts.json", FIFO)
        summary = result["results"][0]["summary"]
        kinds = summary["by_kind"]
        self.assertEqual(sum(stats["count"] for stats in kinds.values()), len(self.data["jobs"]))
        weighted = sum(stats["mean_latency_ms"] * stats["count"] for stats in kinds.values()) / len(self.data["jobs"])
        self.assertAlmostEqual(weighted, summary["mean_latency_ms"])
        self.assertIn("retake (5)", runner.format_result(result))

    def test_retake_first_does_not_starve_segments(self):
        sys.path.insert(0, str(SCHEDULERS))
        import retake_first

        old = {"id": "s", "kind": "script_segment", "arrival_ms": 0}
        new = {"id": "r", "kind": "retake", "arrival_ms": 700000}
        self.assertIs(retake_first.select([old, new], 700000), old)
        fresh = {"id": "r2", "kind": "retake", "arrival_ms": 1000}
        self.assertIs(retake_first.select([old, fresh], 1000), fresh)


class CalibrateTests(TempHome):
    def test_recovers_coefficients(self):
        from gpu_bench import calibrate

        jobs = [{"id": f"j{i}", "arrival_ms": 0, "input": {"text": "x" * (10 + 7 * i)}} for i in range(30)]
        records = [{"id": job["id"], "service_ms": 50 + 2.5 * len(job["input"]["text"])} for job in jobs]
        result = {
            "mode": "live", "created": "now", "gpus": [{"name": "Test GPU"}],
            "workload": {"name": "w", "jobs": jobs},
            "model": {"name": "m", "runtime": {"base_ms": 1, "per_character_ms": 1, "text_field": "text"}},
            "results": [{"name": "fifo", "summary": {"mean_latency_ms": 1}, "jobs": records}],
        }
        fitted = calibrate.fit(result)
        self.assertAlmostEqual(fitted["base_ms"], 50, places=6)
        self.assertAlmostEqual(fitted["per_unit_ms"], 2.5, places=6)
        self.assertAlmostEqual(fitted["r2"], 1.0)
        model = calibrate.calibrated_model(result, fitted)
        self.assertEqual(model["runtime"]["per_character_ms"], 2.5)
        self.assertEqual(model["name"], "m-calibrated")

    def test_rejects_simulations(self):
        from gpu_bench import calibrate

        with self.assertRaisesRegex(ValueError, "live"):
            calibrate.fit({"mode": "simulate"})


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


class PackagedAppTests(TempHome):
    def test_entry_runs_the_command_line_with_arguments(self):
        from contextlib import redirect_stdout

        import launch_app

        out = io.StringIO()
        with redirect_stdout(out):
            self.assertEqual(launch_app.main(["list", "schedulers"]), 0)
        self.assertIn("fair_share", out.getvalue())

    def test_bundle_has_every_module_the_server_needs(self):
        bundle, _args = remote.build_bundle()
        with tarfile.open(fileobj=io.BytesIO(bundle), mode="r:gz") as tar:
            names = set(tar.getnames())
        for module in ("cli", "runner", "compare", "catalog", "backends", "gpus", "store", "remote"):
            self.assertIn(f"gpu_bench/{module}.py", names)


class ChartTests(unittest.TestCase):
    def test_axis_reaches_the_largest_value(self):
        try:
            from gpu_bench.charts import nice_ticks
        except ImportError:  # charts need PySide6 (and its system libraries), which CI doesn't install
            self.skipTest("PySide6 is not available")
        for top in (1, 7, 999, 1056, 1276, 17900, 22337, 24790, 160532):
            ticks = nice_ticks(top)
            self.assertEqual(ticks[0], 0)
            self.assertGreaterEqual(ticks[-1], top)
            self.assertLess(ticks[-2], top)


if __name__ == "__main__":
    unittest.main()
