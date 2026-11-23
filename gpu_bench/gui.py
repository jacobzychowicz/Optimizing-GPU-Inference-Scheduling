"""Desktop window. The command line does not import this module."""

import sys
import threading
import traceback
from pathlib import Path

try:
    from PySide6.QtCore import QThread, Qt, QTimer, Signal
    from PySide6.QtGui import QColor, QPainter, QPen
    from PySide6.QtWidgets import (
        QAbstractItemView,
        QApplication,
        QButtonGroup,
        QCheckBox,
        QComboBox,
        QDialog,
        QDialogButtonBox,
        QFileDialog,
        QFormLayout,
        QGroupBox,
        QHBoxLayout,
        QHeaderView,
        QLabel,
        QLineEdit,
        QListWidget,
        QListWidgetItem,
        QMainWindow,
        QMenu,
        QMessageBox,
        QPlainTextEdit,
        QProgressBar,
        QPushButton,
        QRadioButton,
        QScrollArea,
        QSpinBox,
        QSplitter,
        QTableWidget,
        QTableWidgetItem,
        QTabWidget,
        QToolButton,
        QToolTip,
        QVBoxLayout,
        QWidget,
    )
except ImportError:
    Qt = None


_BAR_COLORS = ("#4c78a8", "#f58518", "#54a24b", "#e45756", "#72b7b2", "#b279a2", "#eeca3b", "#9d755d")
LOCAL = "__local__"


def main():
    if Qt is None:
        print(
            "error: the desktop window needs PySide6.\nInstall it with: pip install PySide6",
            file=sys.stderr,
        )
        return 1
    app = QApplication(sys.argv)
    app.setApplicationName("GPU Inference Scheduler")
    icon = Path(__file__).resolve().parent / "assets" / "icon.png"
    if icon.is_file():
        from PySide6.QtGui import QIcon

        app.setWindowIcon(QIcon(str(icon)))
    window = Window()
    window.resize(1200, 760)
    window.show()
    return app.exec()


if Qt is not None:

    class Task(QThread):
        """Runs fn(emit, cancel) off the UI thread. Events arrive through the event signal."""

        event = Signal(object)
        done = Signal(object)
        failed = Signal(str)
        needs_trust = Signal(object)

        def __init__(self, fn, parent=None):
            super().__init__(parent)
            self.fn = fn
            self.cancel = threading.Event()

        def run(self):
            from gpu_bench.remote import UnknownHostKey
            from gpu_bench.runner import Cancelled

            try:
                result = self.fn(self.event.emit, self.cancel)
            except UnknownHostKey as exc:
                self.needs_trust.emit(exc)
            except Cancelled:
                self.failed.emit("Stopped.")
            except (OSError, ValueError, RuntimeError, KeyError) as exc:
                self.failed.emit(str(exc))
            except Exception:  # show anything else instead of losing it in a thread
                self.failed.emit(traceback.format_exc())
            else:
                self.done.emit(result)

    class PasswordDialog(QDialog):
        """Asks for an SSH password, with an option to keep it in the credential store."""

        def __init__(self, parent, host, remember_default=False):
            super().__init__(parent)
            from gpu_bench.store import can_save_passwords, describe_host

            self.setWindowTitle("SSH password")
            self.password = QLineEdit()
            self.password.setEchoMode(QLineEdit.EchoMode.Password)
            self.remember = QCheckBox("Remember this password")
            if can_save_passwords():
                self.remember.setChecked(remember_default)
                self.remember.setToolTip("Kept in your system's credential store (Windows Credential Manager), not in a file.")
            else:
                self.remember.setEnabled(False)
                self.remember.setToolTip("No credential store is available on this computer.")
            buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
            buttons.accepted.connect(self.accept)
            buttons.rejected.connect(self.reject)
            layout = QVBoxLayout(self)
            layout.addWidget(QLabel(f"Password for <b>{describe_host(host)}</b>"))
            layout.addWidget(self.password)
            layout.addWidget(self.remember)
            layout.addWidget(buttons)
            self.password.setFocus()

    def ask_password(parent, host):
        """(password, remember) or None. Matches remote.with_password's ask()."""
        dialog = PasswordDialog(parent, host, bool(host.get("save_password")))
        if dialog.exec() != QDialog.DialogCode.Accepted or not dialog.password.text():
            return None
        return dialog.password.text(), dialog.remember.isChecked()

    def prepare_host(parent, host):
        """The host with its password attached, asking if needed. None if the user cancelled."""
        if host is None:
            return None
        from gpu_bench.remote import uses_password, with_password
        from gpu_bench.runner import Cancelled

        # A typed password is reused for the rest of the session (only in memory).
        session = getattr(parent, "session_passwords", None)
        if session is not None and uses_password(host) and host["name"] in session:
            return dict(host, password=session[host["name"]], password_source="typed")
        try:
            prepared = with_password(host, lambda h: ask_password(parent, h))
            if session is not None and prepared.get("password_source") == "typed":
                session[host["name"]] = prepared["password"]
            return prepared
        except Cancelled:
            return False
        except ValueError as exc:
            QMessageBox.warning(parent, "Password not saved", str(exc))
            return False

    def confirm_trust(parent, exc):
        """Show a new server's fingerprint. True if the user trusts it (and it is saved)."""
        from gpu_bench.remote import trust_host_key

        box = QMessageBox(parent)
        box.setIcon(QMessageBox.Icon.Question)
        box.setWindowTitle("New server")
        box.setText(str(exc))
        box.setInformativeText("Trust this server and connect?")
        box.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        box.setDefaultButton(QMessageBox.StandardButton.No)
        if box.exec() != QMessageBox.StandardButton.Yes:
            return False
        trust_host_key(exc)
        return True

    class NewWorkloadDialog(QDialog):
        """Generate a workload file into ~/.gpu_bench/workloads."""

        def __init__(self, parent, models, model_name=None):
            super().__init__(parent)
            self.setWindowTitle("New workload")
            self.created = None
            self.models = models
            self.name = QLineEdit("narration-busy")
            self.preset = QComboBox()
            self.preset.addItem("Narration: scripts + retakes", "narration")
            self.preset.addItem("Traffic: steady / bursty / ramp", "traffic")
            self.model = QComboBox()
            for item in models:
                self.model.addItem(item["name"])
            default = model_name if model_name and "tts" in model_name else "example-tts"
            if self.model.findText(default) >= 0:
                self.model.setCurrentText(default)
            self.seed = _spin(1, 999999, 7)

            self.users = _spin(1, 100, 4)
            self.script_users = _spin(1, 100, 2)
            self.seg_min = _spin(1, 5000, 100)
            self.seg_max = _spin(1, 5000, 400)
            self.retakes = _spin(0, 5000, 20)
            self.count = _spin(1, 100000, 200)
            self.rate = _spin(1, 1000, 4)
            self.pattern = QComboBox()
            self.pattern.addItems(["steady", "bursty", "ramp"])
            self.field = QComboBox()
            self.field.addItem("max_tokens (language model)", "max_tokens")
            self.field.addItem("text (speech)", "text")

            form = QFormLayout()
            form.addRow("Name", self.name)
            form.addRow("Kind", self.preset)
            form.addRow("Model", self.model)
            self.narration_rows = [("Users sharing the GPU", self.users), ("Users with a full script", self.script_users),
                                   ("Lines per script, min", self.seg_min), ("Lines per script, max", self.seg_max),
                                   ("Single-line retakes", self.retakes)]
            self.traffic_rows = [("Requests", self.count), ("Requests per second", self.rate),
                                 ("Arrivals", self.pattern), ("Size", self.field)]
            for label, widget in self.narration_rows + self.traffic_rows:
                form.addRow(label, widget)
            form.addRow("Seed", self.seed)
            self.form = form
            self.preset.currentIndexChanged.connect(self._preset_changed)
            self._preset_changed()

            note = QLabel("Narration: each script arrives at once as one request per sentence, while other users "
                          "wait on single-line retakes. Saved to ~/.gpu_bench/workloads.")
            note.setWordWrap(True)
            note.setStyleSheet("color: gray")
            buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
            buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Create")
            buttons.accepted.connect(self._create)
            buttons.rejected.connect(self.reject)
            layout = QVBoxLayout(self)
            layout.addLayout(form)
            layout.addWidget(note)
            layout.addWidget(buttons)

        def _preset_changed(self, *_args):
            narration = self.preset.currentData() == "narration"
            for _label, widget in self.narration_rows:
                self.form.setRowVisible(widget, narration)
            for _label, widget in self.traffic_rows:
                self.form.setRowVisible(widget, not narration)
            self.adjustSize()

        def _create(self):
            import json

            from gpu_bench import workloads
            from gpu_bench.compare import load_json
            from gpu_bench.store import SAFE_NAME, user_dir

            name = self.name.text().strip()
            if not SAFE_NAME.match(name):
                QMessageBox.warning(self, "Bad name", "Use letters, digits, dot, dash, and underscore.")
                return
            path = user_dir("workloads") / f"{name}.json"
            if path.exists() and QMessageBox.question(self, "Replace", f"{name} exists. Replace it?") != QMessageBox.StandardButton.Yes:
                return
            item = next((m for m in self.models if m["name"] == self.model.currentText()), None)
            model = load_json(item["path"]) if item else None
            model_name = item["name"] if item else None
            try:
                if self.preset.currentData() == "narration":
                    data = workloads.generate_narration(
                        name, self.users.value(), self.script_users.value(),
                        (self.seg_min.value(), self.seg_max.value()), self.retakes.value(),
                        model=model, model_name=model_name, seed=self.seed.value(),
                    )
                else:
                    data = workloads.generate_traffic(
                        name, self.count.value(), float(self.rate.value()), self.pattern.currentText(),
                        self.field.currentData(), model=model, model_name=model_name, seed=self.seed.value(),
                    )
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(data, indent=2), encoding="utf-8")
            except (ValueError, OSError) as exc:
                QMessageBox.warning(self, "Can't create workload", str(exc))
                return
            self.created = name
            self.accept()

    def _spin(low, high, value):
        box = QSpinBox()
        box.setRange(low, high)
        box.setValue(value)
        return box

    class Timeline(QWidget):
        """One lane per scheduler. Each bar is a request on the GPU. Hover a bar for details."""

        def __init__(self):
            super().__init__()
            self.results = []
            self.message = "Run a comparison to see the schedule."
            self._bars = []
            self.setMouseTracking(True)
            self.setMinimumHeight(90)

        def set_results(self, results, message=None):
            self.results = results
            self.message = message or "Run a comparison to see the schedule."
            lanes = max(1, len(results))
            self.setMinimumHeight(56 + lanes * 30)
            self.update()

        def paintEvent(self, _event):
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
            self._bars = []
            jobs_all = [job for item in self.results for job in item.get("jobs", [])]
            if not jobs_all:
                painter.setPen(self.palette().color(self.foregroundRole()))
                painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self.message)
                return

            metrics = painter.fontMetrics()
            label_w = max(metrics.horizontalAdvance(item["name"]) for item in self.results) + 16
            makespan = max(job["finish_ms"] for job in jobs_all) or 1
            plot_w = max(1, self.width() - label_w - 12)
            lane_h = 22
            text_pen = QPen(self.palette().color(self.foregroundRole()))
            for index, item in enumerate(self.results):
                y = 8 + index * 30
                painter.setPen(text_pen)
                painter.drawText(4, y, label_w - 12, lane_h,
                                 Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, item["name"])
                for job in item.get("jobs", []):
                    x = label_w + plot_w * job["start_ms"] / makespan
                    width = max(1.0, plot_w * (job["finish_ms"] - job["start_ms"]) / makespan)
                    color = _job_color(job)
                    painter.fillRect(int(x), y, max(1, int(width) - 1), lane_h, color)
                    self._bars.append((int(x), y, int(width), lane_h, item["name"], job))
                    if width >= 48:
                        painter.setPen(QPen(QColor("white")))
                        painter.drawText(int(x) + 4, y, int(width) - 8, lane_h,
                                         Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, str(job["id"]))
            painter.setPen(text_pen)
            end = f"{makespan:.0f} ms"
            painter.drawText(label_w, self.height() - 10, "0 ms")
            painter.drawText(self.width() - metrics.horizontalAdvance(end) - 8, self.height() - 10, end)

        def mouseMoveEvent(self, event):
            pos = event.position().toPoint() if hasattr(event, "position") else event.pos()
            for x, y, w, h, name, job in self._bars:
                if x <= pos.x() <= x + max(w, 2) and y <= pos.y() <= y + h:
                    QToolTip.showText(
                        event.globalPosition().toPoint() if hasattr(event, "globalPosition") else event.globalPos(),
                        f"{name}: {job['id']}\n"
                        f"arrived {job.get('arrival_ms', 0):.0f} ms, started {job['start_ms']:.0f} ms\n"
                        f"waited {job['wait_ms']:.0f} ms, ran {job.get('service_ms', job['finish_ms'] - job['start_ms']):.0f} ms\n"
                        f"latency {job['latency_ms']:.0f} ms",
                        self,
                    )
                    return
            QToolTip.hideText()

    class HostDialog(QDialog):
        """Add, edit, test, and remove SSH hosts."""

        def __init__(self, parent=None, select=None):
            super().__init__(parent)
            self.setWindowTitle("SSH hosts")
            self.resize(760, 460)
            self._task = None
            self._editing = None

            self.list = QListWidget()
            self.list.currentRowChanged.connect(self._load_row)
            new = QPushButton("New")
            new.clicked.connect(self._new)
            delete = QPushButton("Delete")
            delete.clicked.connect(self._delete)
            left = QVBoxLayout()
            left.addWidget(self.list, 1)
            row = QHBoxLayout()
            row.addWidget(new)
            row.addWidget(delete)
            left.addLayout(row)

            self.name = QLineEdit()
            self.host = QLineEdit()
            self.host.setPlaceholderText("gpu.example.com, 10.0.0.5, or a Host alias from ~/.ssh/config")
            self.user = QLineEdit()
            self.user.setPlaceholderText("optional")
            self.port = QLineEdit()
            self.port.setPlaceholderText("22")
            self.key = QLineEdit()
            self.key.setPlaceholderText("optional: uses your ssh agent and default keys")
            browse = QToolButton()
            browse.setText("...")
            browse.clicked.connect(self._browse_key)
            key_row = QHBoxLayout()
            key_row.addWidget(self.key, 1)
            key_row.addWidget(browse)
            self.python = QLineEdit()
            self.python.setPlaceholderText("python3, /opt/venv/bin/python, conda run -n infer python")
            self.workdir = QLineEdit()
            self.workdir.setPlaceholderText(".gpu-bench (inside your home folder)")
            self.options = QLineEdit()
            self.options.setPlaceholderText("optional, comma separated: ProxyJump=bastion, IdentitiesOnly=yes")
            self.auth = QComboBox()
            self.auth.addItem("SSH key / agent", "key")
            self.auth.addItem("Password", "password")
            self.auth.currentIndexChanged.connect(self._auth_changed)
            self.password = QLineEdit()
            self.password.setEchoMode(QLineEdit.EchoMode.Password)
            from gpu_bench.store import can_save_passwords

            self.save_password = QCheckBox("Save password (Windows Credential Manager)")
            if not can_save_passwords():
                self.save_password.setEnabled(False)
                self.save_password.setToolTip("No credential store is available, so the password is asked for each time.")
            self.save_password.toggled.connect(self._auth_changed)

            form = QFormLayout()
            form.addRow("Name", self.name)
            form.addRow("Host", self.host)
            form.addRow("User", self.user)
            form.addRow("Port", self.port)
            form.addRow("Login", self.auth)
            form.addRow("Password", self.password)
            form.addRow("", self.save_password)
            form.addRow("Key file", key_row)
            self.browse = browse
            self.form = form
            self.key_row = key_row
            form.addRow("Python", self.python)
            form.addRow("Scratch folder", self.workdir)
            form.addRow("ssh options", self.options)

            save = QPushButton("Save")
            save.clicked.connect(self._save)
            self.test = QPushButton("Test connection")
            self.test.clicked.connect(self._test)
            buttons = QHBoxLayout()
            buttons.addWidget(save)
            buttons.addWidget(self.test)
            buttons.addStretch(1)

            self.output = QPlainTextEdit()
            self.output.setReadOnly(True)
            self.output.setPlaceholderText(
                "SSH key / agent uses your normal ssh setup (keys, agent, ~/.ssh/config). "
                "Password asks for the password when you connect, or uses the saved one. "
                "The server only needs Python 3.8+."
            )

            right = QVBoxLayout()
            right.addLayout(form)
            right.addLayout(buttons)
            right.addWidget(self.output, 1)

            close = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
            close.rejected.connect(self.reject)
            body = QHBoxLayout()
            body.addLayout(left, 1)
            body.addLayout(right, 2)
            outer = QVBoxLayout(self)
            outer.addLayout(body, 1)
            outer.addWidget(close)
            self._refresh(select)

        def _refresh(self, select=None):
            from gpu_bench.store import describe_host, load_hosts

            self.hosts = load_hosts()
            self.list.blockSignals(True)
            self.list.clear()
            for host in self.hosts:
                self.list.addItem(f"{host['name']}  ({describe_host(host)})")
            self.list.blockSignals(False)
            names = [host["name"] for host in self.hosts]
            if select in names:
                self.list.setCurrentRow(names.index(select))
            elif self.hosts:
                self.list.setCurrentRow(0)
            else:
                self._new()

        def _load_row(self, row):
            if not 0 <= row < len(self.hosts):
                return
            host = self.hosts[row]
            self._editing = host["name"]
            self.name.setText(host["name"])
            self.host.setText(host["host"])
            self.user.setText(host["user"])
            self.port.setText(str(host["port"] or ""))
            self.key.setText(host["key"])
            self.python.setText(host["python"])
            self.workdir.setText(host["workdir"])
            self.options.setText(", ".join(host["ssh_options"]))
            self.auth.setCurrentIndex(max(0, self.auth.findData(host["auth"])))
            self.save_password.setChecked(host["save_password"])
            self.password.clear()
            self._auth_changed()

        def _new(self):
            self._editing = None
            self.list.clearSelection()
            for field in (self.name, self.host, self.user, self.port, self.key, self.options):
                field.clear()
            self.python.setText("python3")
            self.workdir.setText(".gpu-bench")
            self.auth.setCurrentIndex(0)
            self.save_password.setChecked(False)
            self.password.clear()
            self._auth_changed()
            self.host.setFocus()

        def _auth_changed(self, *_args):
            from gpu_bench.store import get_saved_password

            password = self.auth.currentData() == "password"
            self.form.setRowVisible(self.password, password)
            self.form.setRowVisible(self.save_password, password)
            self.form.setRowVisible(self.key_row, not password)
            self.form.setRowVisible(self.options, not password)
            has_saved = bool(self._editing and get_saved_password(self._editing))
            if not self.save_password.isChecked():
                self.password.setPlaceholderText("used for Test connection only; asked for each time you connect")
            elif has_saved:
                self.password.setPlaceholderText("saved. Type a new one to replace it")
            else:
                self.password.setPlaceholderText("type it to save it, or leave blank to be asked once")

        def _form(self):
            return {
                "name": self.name.text().strip(),
                "host": self.host.text().strip(),
                "user": self.user.text().strip(),
                "port": self.port.text().strip() or None,
                "key": self.key.text().strip(),
                "python": self.python.text().strip(),
                "workdir": self.workdir.text().strip(),
                "ssh_options": [part.strip() for part in self.options.text().split(",") if part.strip()],
                "auth": self.auth.currentData(),
                "save_password": self.save_password.isChecked(),
            }

        def _save(self):
            from gpu_bench.store import save_host, set_saved_password

            form = self._form()
            if form["auth"] == "password":
                form["ssh_options"] = []
                form["key"] = ""
            try:
                host = save_host(form, replace=self._editing)
                typed = self.password.text()
                if host["save_password"] and typed:
                    set_saved_password(host["name"], typed)
            except ValueError as exc:
                QMessageBox.warning(self, "Can't save host", str(exc))
                return None
            self._refresh(host["name"])
            note = ""
            if host["auth"] == "password":
                note = " The password is saved in Windows Credential Manager." if host["save_password"] and typed else (
                    " You'll be asked for the password when connecting." if not host["save_password"]
                    else "")
            self.output.setPlainText(f"Saved {host['name']}.{note}")
            return host

        def _delete(self):
            from gpu_bench.store import remove_host

            if not self._editing:
                return
            if QMessageBox.question(self, "Delete host", f"Forget {self._editing}?") != QMessageBox.StandardButton.Yes:
                return
            remove_host(self._editing)
            self._refresh()

        def _browse_key(self):
            path, _ = QFileDialog.getOpenFileName(self, "Private key", str(Path.home() / ".ssh"))
            if path:
                self.key.setText(path)

        def _test(self):
            from gpu_bench.store import get_saved_password, normalize_host

            try:
                host = normalize_host(self._form())
            except ValueError as exc:
                self.output.setPlainText(str(exc))
                return
            if host["auth"] == "password":
                typed = self.password.text()
                saved = get_saved_password(self._editing) if self._editing and host["save_password"] else None
                if typed:
                    host["password"] = typed
                elif saved:
                    host.update(password=saved, password_source="saved", name=self._editing)
                else:
                    answer = ask_password(self, host)
                    if answer is None:
                        return
                    host["password"] = answer[0]
                    if answer[1]:
                        self.password.setText(answer[0])
                        self.save_password.setChecked(True)
            self._run_test(host)

        def _run_test(self, host):
            from gpu_bench.gpus import format_probe
            from gpu_bench.remote import probe
            from gpu_bench.store import describe_host

            self.output.setPlainText(f"Connecting to {describe_host(host)} ...")
            self.test.setEnabled(False)
            self._task = Task(lambda _emit, _cancel: probe(host), self)
            self._task.done.connect(lambda info: self.output.setPlainText("Connected.\n\n" + format_probe(info)))
            self._task.failed.connect(self.output.setPlainText)
            self._task.needs_trust.connect(lambda exc: self._trust_and_retry(exc, host))
            self._task.finished.connect(lambda: self.test.setEnabled(True))
            self._task.start()

        def _trust_and_retry(self, exc, host):
            if confirm_trust(self, exc):
                self._run_test(host)
            else:
                self.output.setPlainText("Not connected: the server was not trusted.")

        def reject(self):
            if self._task is not None and self._task.isRunning():
                self._task.wait(100)
            super().reject()

    class Window(QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("GPU Inference Scheduler")
            from gpu_bench.store import load_settings

            self.settings = load_settings()
            self.session_passwords = {}
            self.task = None
            self.probe_task = None
            self.result = None
            self.live = {}
            self.progress_total = 0

            central = QWidget()
            self.setCentralWidget(central)
            outer = QVBoxLayout(central)
            outer.addLayout(self._build_machine_bar())
            splitter = QSplitter(Qt.Orientation.Horizontal)
            splitter.addWidget(self._build_setup())
            splitter.addWidget(self._build_tabs())
            splitter.setStretchFactor(0, 1)
            splitter.setStretchFactor(1, 3)
            splitter.setSizes([340, 860])
            outer.addWidget(splitter, 1)

            self._reload_catalog()
            self._reload_hosts(self.settings.get("host"))
            self._reload_history()
            self._mode_changed()

        # layout -------------------------------------------------------

        def _build_machine_bar(self):
            self.machine = QComboBox()
            self.machine.setMinimumWidth(260)
            self.machine.currentIndexChanged.connect(self._machine_changed)
            hosts = QPushButton("Manage hosts...")
            hosts.clicked.connect(self._manage_hosts)
            self.check = QPushButton("Check GPUs")
            self.check.clicked.connect(self._probe)
            self.gpu_label = QLabel("")
            self.gpu_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            bar = QHBoxLayout()
            bar.addWidget(QLabel("Run on"))
            bar.addWidget(self.machine)
            bar.addWidget(hosts)
            bar.addWidget(self.check)
            bar.addWidget(self.gpu_label, 1)
            return bar

        def _build_setup(self):
            mode_box = QGroupBox("Mode")
            self.simulate = QRadioButton("Simulate")
            self.simulate.setToolTip("Compute the schedule from the model's runtime formula. Instant, no GPU needed.")
            self.live_mode = QRadioButton("Live on GPU")
            self.live_mode.setToolTip(
                "Replay the workload on a real clock and run each request on the model's backend.\n"
                "Measures real latency and GPU use."
            )
            group = QButtonGroup(self)
            group.addButton(self.simulate)
            group.addButton(self.live_mode)
            (self.live_mode if self.settings.get("mode") == "live" else self.simulate).setChecked(True)
            self.simulate.toggled.connect(self._mode_changed)
            self.repeats = QSpinBox()
            self.repeats.setRange(1, 50)
            self.repeats.setValue(int(self.settings.get("repeats", 1) or 1))
            self.repeats.setToolTip("Run each scheduler this many times and average the numbers.")
            mode_layout = QHBoxLayout(mode_box)
            mode_layout.addWidget(self.simulate)
            mode_layout.addWidget(self.live_mode)
            mode_layout.addStretch(1)
            mode_layout.addWidget(QLabel("Repeats"))
            mode_layout.addWidget(self.repeats)

            self.models = QListWidget()
            self.workloads = QListWidget()
            self.schedulers = QListWidget()
            self.workloads.currentRowChanged.connect(self._workload_changed)
            self.models.currentRowChanged.connect(self._model_changed)

            all_btn = QToolButton()
            all_btn.setText("All")
            all_btn.clicked.connect(lambda: self._check_all(True))
            none_btn = QToolButton()
            none_btn.setText("None")
            none_btn.clicked.connect(lambda: self._check_all(False))
            sched_head = QHBoxLayout()
            sched_head.addWidget(QLabel("Schedulers"))
            sched_head.addStretch(1)
            sched_head.addWidget(all_btn)
            sched_head.addWidget(none_btn)

            add = QPushButton("Add file...")
            add.setToolTip("A workload or model (.json) or a scheduler (.py). It stays in the lists next time.")
            add.clicked.connect(self._add_file)
            new_workload = QPushButton("New workload...")
            new_workload.setToolTip("Generate narration traffic (scripts + retakes) or generic request traffic.")
            new_workload.clicked.connect(self._new_workload)
            file_row = QHBoxLayout()
            file_row.addWidget(new_workload)
            file_row.addWidget(add)

            self.run_btn = QPushButton("Run")
            self.run_btn.setDefault(True)
            self.run_btn.clicked.connect(self.run_comparison)
            self.stop_btn = QPushButton("Stop")
            self.stop_btn.setEnabled(False)
            self.stop_btn.clicked.connect(self._stop)
            buttons = QHBoxLayout()
            buttons.addWidget(self.run_btn, 2)
            buttons.addWidget(self.stop_btn, 1)
            self.progress = QProgressBar()
            self.progress.setTextVisible(True)
            self.progress.setValue(0)
            self.status = QLabel("Pick a model, a workload, and at least one scheduler.")
            self.status.setWordWrap(True)

            column = QVBoxLayout()
            column.addWidget(mode_box)
            column.addWidget(QLabel("Workload"))
            column.addWidget(self.workloads, 1)
            column.addWidget(QLabel("Model"))
            column.addWidget(self.models, 1)
            self.model_hint = QLabel("")
            self.model_hint.setWordWrap(True)
            self.model_hint.setStyleSheet("color: gray")
            column.addWidget(self.model_hint)
            column.addLayout(sched_head)
            column.addWidget(self.schedulers, 1)
            column.addLayout(file_row)
            column.addLayout(buttons)
            column.addWidget(self.progress)
            column.addWidget(self.status)
            panel = QWidget()
            panel.setLayout(column)
            return panel

        def _build_tabs(self):
            self.tabs = QTabWidget()

            # results
            self.headline = QLabel("")
            self.headline.setWordWrap(True)
            self.table = QTableWidget(0, 0)
            _plain_table(self.table)
            self.note = QLabel("")
            self.note.setWordWrap(True)
            self.timeline = Timeline()
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setWidget(self.timeline)
            export = QPushButton("Export...")
            menu = QMenu(export)
            menu.addAction("Summary as CSV...", lambda: self._export(self.result, "summary"))
            menu.addAction("Every job as CSV...", lambda: self._export(self.result, "jobs"))
            menu.addAction("Full result as JSON...", lambda: self._export(self.result, "json"))
            export.setMenu(menu)
            self.export_btn = export
            export.setEnabled(False)
            self.calibrate_btn = QPushButton("Calibrate model...")
            self.calibrate_btn.setToolTip("Fit the model's runtime formula to the GPU times this live run measured.")
            self.calibrate_btn.setEnabled(False)
            self.calibrate_btn.clicked.connect(self._calibrate)
            head = QHBoxLayout()
            head.addWidget(self.headline, 1)
            head.addWidget(self.calibrate_btn)
            head.addWidget(export)
            results = QWidget()
            layout = QVBoxLayout(results)
            layout.addLayout(head)
            layout.addWidget(self.table, 2)
            layout.addWidget(self.note)
            layout.addWidget(QLabel("Schedule (hover a bar for details)"))
            layout.addWidget(scroll, 2)
            self.tabs.addTab(results, "Results")

            # charts
            from gpu_bench.charts import ALL, LatencyBars, LatencyCDF

            self.bars = LatencyBars()
            self.cdf = LatencyCDF()
            self.chart_kind = QComboBox()
            self.chart_kind.addItem(ALL)
            self.chart_kind.currentTextChanged.connect(self.cdf.set_kind)
            save_png = QPushButton("Save as PNG...")
            save_png.clicked.connect(self._save_charts)
            self.charts_box = QWidget()
            self.charts_box.setAutoFillBackground(True)
            box = QVBoxLayout(self.charts_box)
            box.setContentsMargins(0, 0, 0, 0)
            box.addWidget(self.bars, 1)
            box.addWidget(self.cdf, 1)
            charts = QWidget()
            layout = QVBoxLayout(charts)
            row = QHBoxLayout()
            row.addWidget(QLabel("Distribution for"))
            row.addWidget(self.chart_kind)
            row.addStretch(1)
            row.addWidget(save_png)
            layout.addLayout(row)
            layout.addWidget(self.charts_box, 1)
            self.tabs.addTab(charts, "Charts")

            # jobs
            self.job_scheduler = QComboBox()
            self.job_scheduler.currentIndexChanged.connect(self._show_jobs)
            self.jobs_table = QTableWidget(0, 7)
            self.jobs_table.setHorizontalHeaderLabels(
                ["id", "arrival_ms", "start_ms", "finish_ms", "wait_ms", "service_ms", "latency_ms"]
            )
            _plain_table(self.jobs_table)
            self.jobs_table.setSortingEnabled(True)
            jobs = QWidget()
            layout = QVBoxLayout(jobs)
            row = QHBoxLayout()
            row.addWidget(QLabel("Scheduler"))
            row.addWidget(self.job_scheduler, 1)
            layout.addLayout(row)
            layout.addWidget(self.jobs_table, 1)
            self.tabs.addTab(jobs, "Jobs")

            # log
            self.log = QPlainTextEdit()
            self.log.setReadOnly(True)
            self.log.setMaximumBlockCount(5000)
            self.tabs.addTab(self.log, "Log")

            # history
            self.history = QListWidget()
            self.history.itemDoubleClicked.connect(lambda _item: self._open_history())
            open_btn = QPushButton("Open")
            open_btn.clicked.connect(self._open_history)
            hist_export = QPushButton("Export...")
            hist_menu = QMenu(hist_export)
            hist_menu.addAction("Summary as CSV...", lambda: self._export(self._history_result(), "summary"))
            hist_menu.addAction("Every job as CSV...", lambda: self._export(self._history_result(), "jobs"))
            hist_menu.addAction("Full result as JSON...", lambda: self._export(self._history_result(), "json"))
            hist_export.setMenu(hist_menu)
            delete = QPushButton("Delete")
            delete.clicked.connect(self._delete_history)
            refresh = QPushButton("Refresh")
            refresh.clicked.connect(self._reload_history)
            history = QWidget()
            layout = QVBoxLayout(history)
            layout.addWidget(self.history, 1)
            row = QHBoxLayout()
            for button in (open_btn, hist_export, delete, refresh):
                row.addWidget(button)
            row.addStretch(1)
            layout.addLayout(row)
            self.tabs.addTab(history, "History")
            return self.tabs

        # catalog ------------------------------------------------------

        def _reload_catalog(self, keep=True):
            from gpu_bench.catalog import item_from_file, model_items, scheduler_items, workload_items

            previous = (self._name_of(self.workloads, "_workloads"), self._name_of(self.models, "_models")) if keep else (None, None)
            checked = {item["name"] for item in _checked(self.schedulers, getattr(self, "_schedulers", []))} if keep else set()
            self._models = model_items()
            self._workloads = workload_items()
            self._schedulers = scheduler_items()
            for path in self.settings.get("extra_files", []):
                try:
                    item = item_from_file(path)
                except (ValueError, OSError):
                    continue
                target = {"model": self._models, "workload": self._workloads, "scheduler": self._schedulers}[item["kind"]]
                if all(Path(existing["path"]).resolve() != Path(path).resolve() for existing in target):
                    item["name"] = item["name"] if all(x["name"] != item["name"] for x in target) else f"{item['name']} ({Path(path).parent.name})"
                    target.append(item)
            for item in self._models:
                item["data"] = _safe_json(item["path"])

            self.workloads.blockSignals(True)
            self.models.blockSignals(True)
            _fill_plain(self.workloads, self._workloads, lambda item: item.get("description", ""))
            _fill_plain(self.models, self._models, _model_tooltip, _model_label)
            _fill_checks(self.schedulers, self._schedulers, checked)
            self.workloads.blockSignals(False)
            self.models.blockSignals(False)

            names = [item["name"] for item in self._workloads]
            self.workloads.setCurrentRow(names.index(previous[0]) if previous[0] in names else (0 if names else -1))
            if previous[1]:
                model_names = [item["name"] for item in self._models]
                if previous[1] in model_names:
                    self.models.setCurrentRow(model_names.index(previous[1]))

        def _name_of(self, widget, attr):
            items = getattr(self, attr, None)
            if not items:
                return None
            row = widget.currentRow()
            return items[row]["name"] if 0 <= row < len(items) else None

        def _workload_changed(self, row):
            if not 0 <= row < len(self._workloads):
                return
            paired = self._workloads[row].get("model")
            names = [item["name"] for item in self._models]
            if paired in names:
                self.models.setCurrentRow(names.index(paired))
            elif self.models.currentRow() < 0 and names:
                self.models.setCurrentRow(0)

        def _model_changed(self, row):
            if not 0 <= row < len(self._models):
                self.model_hint.setText("")
                return
            from gpu_bench.backends import describe

            data = self._models[row].get("data") or {}
            self.model_hint.setText(f"Live runs use: {describe(data)}")

        def _check_all(self, state):
            for index in range(self.schedulers.count()):
                self.schedulers.item(index).setCheckState(Qt.CheckState.Checked if state else Qt.CheckState.Unchecked)

        def _add_file(self):
            from gpu_bench.catalog import item_from_file

            path, _ = QFileDialog.getOpenFileName(
                self, "Add a workload, model, or scheduler", str(Path.cwd()),
                "gpu-bench files (*.json *.py);;All files (*)",
            )
            if not path:
                return
            try:
                item = item_from_file(path)
            except (ValueError, OSError) as exc:
                QMessageBox.warning(self, "Can't use that file", str(exc))
                return
            extra = self.settings.setdefault("extra_files", [])
            if path not in extra:
                extra.append(path)
                self._save_settings()
            self._reload_catalog()
            self.status.setText(f"Added {item['kind']} {item['name']}.")

        def _new_workload(self):
            dialog = NewWorkloadDialog(self, self._models, self._current_model_name())
            if dialog.exec() != QDialog.DialogCode.Accepted or dialog.created is None:
                return
            self._reload_catalog()
            names = [item["name"] for item in self._workloads]
            if dialog.created in names:
                self.workloads.setCurrentRow(names.index(dialog.created))
            self.status.setText(f"Created workload {dialog.created}.")

        def _current_model_name(self):
            model = _current(self.models, self._models)
            return model["name"] if model else None

        def _calibrate(self):
            import json

            from gpu_bench import calibrate
            from gpu_bench.store import user_dir

            if not self.result:
                return
            try:
                coefficients = calibrate.fit(self.result)
                model = calibrate.calibrated_model(self.result, coefficients)
                rows = calibrate.check(self.result, model)
            except (ValueError, KeyError) as exc:
                QMessageBox.warning(self, "Can't calibrate", str(exc))
                return
            box = QMessageBox(self)
            box.setWindowTitle("Calibrated model")
            box.setText(f"Save {model['name']} so simulations behave like this hardware?")
            box.setInformativeText(calibrate.format_report(coefficients, rows))
            box.setStyleSheet("QLabel { font-family: Consolas, 'Courier New', monospace; }")
            box.setStandardButtons(QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Cancel)
            if box.exec() != QMessageBox.StandardButton.Save:
                return
            path = user_dir("models") / f"{model['name']}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(model, indent=2), encoding="utf-8")
            self._reload_catalog()
            self.status.setText(f"Saved {model['name']}. Pick it and Simulate to test policies in seconds.")

        def _save_charts(self):
            if not self.result:
                QMessageBox.information(self, "Nothing to save", "Run a comparison first.")
                return
            name = self.result.get("workload", {}).get("name", "run")
            path, _ = QFileDialog.getSaveFileName(self, "Save charts", str(Path.home() / f"{name}-charts.png"), "PNG (*.png)")
            if not path:
                return
            if not self.charts_box.grab().save(path, "PNG"):
                QMessageBox.warning(self, "Save failed", f"Could not write {path}")
                return
            self.status.setText(f"Wrote {path}")

        # machine ------------------------------------------------------

        def _reload_hosts(self, select=None):
            from gpu_bench.store import describe_host, load_hosts

            self.hosts = {host["name"]: host for host in load_hosts()}
            self.machine.blockSignals(True)
            self.machine.clear()
            self.machine.addItem("This computer", LOCAL)
            for name, host in self.hosts.items():
                self.machine.addItem(f"{name}  ({describe_host(host)})  over SSH", name)
            index = self.machine.findData(select) if select else 0
            self.machine.setCurrentIndex(max(index, 0))
            self.machine.blockSignals(False)
            self._machine_changed()

        def _current_host(self):
            key = self.machine.currentData()
            return None if key in (None, LOCAL) else self.hosts.get(key)

        def _machine_changed(self, *_args):
            host = self._current_host()
            self.settings["host"] = host["name"] if host else None
            self._save_settings()
            if host is None:
                self._probe()
            else:
                self.gpu_label.setText("Press Check GPUs to connect and see this server's GPUs.")

        def _manage_hosts(self):
            current = self._current_host()
            dialog = HostDialog(self, current["name"] if current else None)
            dialog.exec()
            self.session_passwords.clear()  # passwords may have changed in the dialog
            self._reload_hosts(self.settings.get("host"))

        def _probe(self):
            if self.probe_task is not None and self.probe_task.isRunning():
                return
            host = prepare_host(self, self._current_host())
            if host is False:
                return
            if host is None:
                from gpu_bench.gpus import probe

                fn = lambda _emit, _cancel: probe(check_torch=True)
                self.gpu_label.setText("Checking this computer...")
            else:
                from gpu_bench.remote import probe as remote_probe
                from gpu_bench.store import describe_host

                fn = lambda _emit, _cancel: remote_probe(host)
                self.gpu_label.setText(f"Connecting to {describe_host(host)}...")
            self.check.setEnabled(False)
            task = Task(fn, self)
            task.done.connect(self._probe_done)
            task.failed.connect(self._probe_failed)
            task.needs_trust.connect(lambda exc: self._trust_then(exc, self._probe, "Not connected: server not trusted."))
            task.finished.connect(lambda: self.check.setEnabled(True))
            self.probe_task = task
            task.start()

        def _probe_done(self, info):
            from gpu_bench.gpus import format_probe, short_gpu_label

            torch = info.get("torch") or {}
            extra = ""
            if torch.get("installed") and not torch.get("error"):
                extra = f"   PyTorch {torch.get('version')}" + (" (CUDA)" if torch.get("cuda") else "")
            self.gpu_label.setText(f"{info.get('machine', '')}: {short_gpu_label(info)}{extra}")
            self.gpu_label.setToolTip(format_probe(info))
            self._log(format_probe(info))

        def _forget_session_password(self, message):
            host = self._current_host()
            if host and "rejected the user name or password" in message:
                self.session_passwords.pop(host["name"], None)

        def _probe_failed(self, message):
            self._forget_session_password(message)
            self.gpu_label.setText("Couldn't check GPUs. See the Log tab.")
            self._log(message)

        # running ------------------------------------------------------

        def _mode(self):
            return "live" if self.live_mode.isChecked() else "simulate"

        def _mode_changed(self, *_args):
            self.repeats.setEnabled(self._mode() == "live")
            self.settings["mode"] = self._mode()
            self._save_settings()

        def run_comparison(self):
            model = _current(self.models, self._models)
            workload = _current(self.workloads, self._workloads)
            chosen = _checked(self.schedulers, self._schedulers)
            missing = []
            if model is None:
                missing.append("a model")
            if workload is None:
                missing.append("a workload")
            if not chosen:
                missing.append("at least one scheduler")
            if missing:
                self.status.setText("Still needed: " + ", ".join(missing) + ".")
                return

            mode = self._mode()
            repeats = self.repeats.value() if mode == "live" else 1
            host = prepare_host(self, self._current_host())
            if host is False:
                self.status.setText("Run cancelled.")
                return
            self.settings["repeats"] = self.repeats.value()
            self._save_settings()
            schedulers = [(item["name"], item["path"]) for item in chosen]
            job_count = len((_safe_json(workload["path"]) or {}).get("jobs") or [])
            self.progress_total = max(1, job_count * len(schedulers) * repeats)
            self.progress.setRange(0, self.progress_total if mode == "live" or host else 0)
            self.progress.setValue(0)
            self.live = {}
            self.timeline.set_results([], "Starting...")
            self.table.setRowCount(0)
            self.note.setText("")
            self.headline.setText(f"Running <b>{workload['name']}</b> on <b>{model['name']}</b> ...")
            self.export_btn.setEnabled(False)
            self.tabs.setCurrentIndex(0)

            from gpu_bench.cli import run_anywhere

            def work(emit, cancel):
                return run_anywhere(host, workload["path"], model["path"], schedulers, mode, repeats, emit, cancel)

            where = host["name"] if host else "this computer"
            self._log(f"--- {workload['name']} on {model['name']}, {mode}, {where}")
            self.status.setText(f"Running {workload['name']} on {where}...")
            self.task = Task(work, self)
            self.task.event.connect(self._on_event)
            self.task.done.connect(self._run_done)
            self.task.failed.connect(self._run_failed)
            self.task.needs_trust.connect(
                lambda exc: self._trust_then(exc, self.run_comparison, "Run cancelled: server not trusted.")
            )
            self.task.finished.connect(self._run_finished)
            self.run_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.task.start()

        def _on_event(self, event):
            kind = event.get("event")
            if kind == "log":
                self._log(event["message"])
            elif kind == "scheduler_start":
                suffix = f" (run {event['repeat']} of {event['repeats']})" if event.get("repeats", 1) > 1 else ""
                self.status.setText(f"Running {event['scheduler']}{suffix}...")
                self._log(f"{event['scheduler']}{suffix}")
                self.live[event["scheduler"]] = []
                self._draw_live()
            elif kind == "job_start":
                suffix = f" (run {event['repeat']})" if event.get("repeat", 1) > 1 else ""
                self.status.setText(
                    f"{event['scheduler']}{suffix}: running {event['id']} ({event['done'] + 1} of {event['total']})..."
                )
                self._draw_live(f"{event['scheduler']} is running {event['id']} on the GPU...")
            elif kind == "job":
                job = event["job"]
                self.live.setdefault(event["scheduler"], []).append(job)
                self.progress.setValue(min(self.progress_total, self.progress.value() + 1))
                self._log(f"  {event['done']}/{event['total']} {job['id']}: waited {job['wait_ms']:.0f} ms, ran {job['service_ms']:.0f} ms")
                self._draw_live()
            elif kind == "scheduler_done":
                summary = event["summary"]
                self._log(f"  {event['scheduler']}: mean {summary['mean_latency_ms']:.1f} ms, p95 {summary['p95_latency_ms']:.1f} ms")

        def _draw_live(self, message=None):
            self.timeline.set_results(
                [{"name": name, "jobs": jobs} for name, jobs in self.live.items()],
                message or "Waiting for the first request to finish...",
            )

        def _trust_then(self, exc, retry, refused):
            self.progress.setRange(0, 1)
            self.progress.setValue(0)
            if confirm_trust(self, exc):
                # let the finished task release its buttons before starting again
                QTimer.singleShot(0, retry)
            else:
                self.status.setText(refused)
                if retry == self._probe:
                    self.gpu_label.setText(refused)

        def _run_done(self, result):
            from gpu_bench.store import save_run

            self.progress.setRange(0, 1)
            self.progress.setValue(1)
            try:
                path = save_run(result)
                self._log(f"Saved to {path}")
            except (OSError, ValueError) as exc:
                self._log(f"Could not save to history: {exc}")
            self._show(result)
            self._reload_history()
            self.status.setText(f"Done: {result['workload']['name']} on {result['model']['name']} ({result['mode']}).")

        def _run_failed(self, message):
            self._forget_session_password(message)
            self.progress.setRange(0, 1)
            self.progress.setValue(0)
            self._log(message)
            first = message.strip().splitlines()[0] if message.strip() else "failed"
            self.status.setText(f"{first}  (details in the Log tab)" if "\n" in message.strip() else first)
            if message != "Stopped.":
                self.tabs.setCurrentWidget(self.log)

        def _run_finished(self):
            self.run_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)

        def _stop(self):
            if self.task is not None and self.task.isRunning():
                self.task.cancel.set()
                self.status.setText("Stopping after the current request...")
                self.stop_btn.setEnabled(False)

        # showing results ----------------------------------------------

        def _show(self, result):
            from gpu_bench.runner import lowest

            self.result = result
            self.export_btn.setEnabled(True)
            rows = result["results"]
            gpu_cols = any("gpu_util_pct" in item["summary"] for item in rows)
            spread = any("mean_latency_spread_ms" in item["summary"] for item in rows)
            headers = ["scheduler", "mean_ms", "p95_ms", "wait_ms", "jobs/s"]
            keys = [None, "mean_latency_ms", "p95_latency_ms", "mean_wait_ms", "jobs_per_s"]
            if spread:
                headers.append("± mean_ms")
                keys.append("mean_latency_spread_ms")
            if gpu_cols:
                headers += ["gpu busy %", "gpu mem MB"]
                keys += ["gpu_util_pct", "gpu_mem_peak_mb"]
            self.table.clear()
            self.table.setColumnCount(len(headers))
            self.table.setHorizontalHeaderLabels(headers)
            table_rows = []
            for item in rows:
                table_rows.append((item["name"], item["summary"], False))
                for kind, stats in (item["summary"].get("by_kind") or {}).items():
                    table_rows.append((f"    {kind} ({stats['count']:.0f})", stats, True))
            self.table.setRowCount(len(table_rows))
            best_mean = lowest(rows, "mean_latency_ms")[1] if len(rows) > 1 else []
            dim = self.palette().color(self.foregroundRole())
            dim.setAlpha(150)
            for row, (name, summary, sub) in enumerate(table_rows):
                for column, key in enumerate(keys):
                    if key is None:
                        cell = QTableWidgetItem(name)
                        if name in best_mean and not sub:
                            font = cell.font()
                            font.setBold(True)
                            cell.setFont(font)
                    else:
                        value = summary.get(key)
                        if value is None:
                            text = "-"
                        elif key == "jobs_per_s":
                            text = f"{value:.2f}"
                        elif key in ("gpu_util_pct", "gpu_mem_peak_mb"):
                            text = f"{value:.0f}"
                        else:
                            text = f"{value:.1f}"
                        cell = QTableWidgetItem("" if sub and value is None else text)
                        cell.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                    if sub:
                        cell.setForeground(dim)
                    self.table.setItem(row, column, cell)
            self.table.resizeColumnsToContents()
            self.calibrate_btn.setEnabled(result.get("mode") == "live")
            self.bars.set_result(result)
            self.cdf.set_result(result)
            from gpu_bench.charts import ALL, kinds_of

            current = self.chart_kind.currentText()
            self.chart_kind.blockSignals(True)
            self.chart_kind.clear()
            self.chart_kind.addItems([ALL, *kinds_of(result)])
            self.chart_kind.setCurrentText(current if current in [ALL, *kinds_of(result)] else ALL)
            self.chart_kind.blockSignals(False)
            self.cdf.set_kind(self.chart_kind.currentText())

            where = result.get("target", "local")
            where = "this computer" if where == "local" else where
            gpus = ", ".join(gpu["name"] for gpu in result.get("gpus") or [])
            bits = [
                f"<b>{result['workload']['name']}</b> on <b>{result['model']['name']}</b>",
                f"{result.get('mode', 'simulate')} on {where}" + (f" ({result.get('machine')})" if result.get("machine") else ""),
            ]
            if gpus:
                bits.append(gpus)
            if result.get("repeats", 1) > 1:
                bits.append(f"average of {result['repeats']} runs")
            if result.get("created"):
                bits.append(result["created"].replace("T", " "))
            self.headline.setText(" · ".join(bits))

            notes = []
            if len(rows) > 1:
                for key, label in (("mean_latency_ms", "mean latency"), ("p95_latency_ms", "p95 latency"),
                                   ("mean_wait_ms", "mean wait")):
                    best, names = lowest(rows, key)
                    notes.append(f"Lowest {label}: {', '.join(names)} ({best:.1f} ms)")
                for kind in sorted({k for item in rows for k in (item["summary"].get("by_kind") or {})}):
                    scored = [(item["summary"]["by_kind"][kind]["mean_latency_ms"], item["name"])
                              for item in rows if kind in (item["summary"].get("by_kind") or {})]
                    best, worst = min(scored), max(scored)
                    factor = f", {worst[0] / best[0]:.1f}x faster than {worst[1]}" if best[0] > 0 and worst[0] > best[0] * 1.05 else ""
                    notes.append(f"Lowest mean latency for {kind}: {best[1]} ({best[0]:.1f} ms{factor})")
            notes.append("mean_ms and p95_ms are latency from arrival to finish. wait_ms is time spent waiting.")
            self.note.setText("\n".join(notes))
            self.timeline.set_results(rows)

            self.job_scheduler.blockSignals(True)
            self.job_scheduler.clear()
            for item in rows:
                self.job_scheduler.addItem(item["name"])
            self.job_scheduler.blockSignals(False)
            self._show_jobs()

        def _show_jobs(self, *_args):
            if not self.result:
                return
            index = self.job_scheduler.currentIndex()
            rows = self.result["results"]
            if not 0 <= index < len(rows):
                self.jobs_table.setRowCount(0)
                return
            jobs = rows[index]["jobs"]
            keys = ["id", "arrival_ms", "start_ms", "finish_ms", "wait_ms", "service_ms", "latency_ms"]
            self.jobs_table.setSortingEnabled(False)
            self.jobs_table.setRowCount(len(jobs))
            for row, job in enumerate(jobs):
                for column, key in enumerate(keys):
                    value = job.get(key, "")
                    cell = QTableWidgetItem()
                    if key == "id":
                        cell.setText(str(value))
                    else:
                        cell.setData(Qt.ItemDataRole.DisplayRole, round(float(value), 1) if value != "" else "")
                        cell.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                    self.jobs_table.setItem(row, column, cell)
            self.jobs_table.setSortingEnabled(True)
            self.jobs_table.resizeColumnsToContents()

        # history ------------------------------------------------------

        def _reload_history(self):
            from gpu_bench.store import list_runs

            self.history.clear()
            for row in list_runs():
                where = "this computer" if row["target"] == "local" else row["target"]
                text = (
                    f"{row['created'].replace('T', ' ')}   {row['workload']} on {row['model']}   "
                    f"{row['mode']} · {where}   best mean: {row['best']}"
                )
                item = QListWidgetItem(text)
                item.setData(Qt.ItemDataRole.UserRole, str(row["path"]))
                item.setToolTip(", ".join(row["schedulers"]))
                self.history.addItem(item)
            if not self.history.count():
                placeholder = QListWidgetItem("Finished runs show up here.")
                placeholder.setFlags(Qt.ItemFlag.NoItemFlags)
                self.history.addItem(placeholder)

        def _history_path(self):
            item = self.history.currentItem()
            return item.data(Qt.ItemDataRole.UserRole) if item else None

        def _history_result(self):
            from gpu_bench.store import load_run

            path = self._history_path()
            if not path:
                return None
            try:
                return load_run(path)
            except ValueError as exc:
                QMessageBox.warning(self, "Can't open run", str(exc))
                return None

        def _open_history(self):
            result = self._history_result()
            if result:
                self._show(result)
                self.tabs.setCurrentIndex(0)

        def _delete_history(self):
            from gpu_bench.store import delete_run

            path = self._history_path()
            if not path:
                return
            if QMessageBox.question(self, "Delete run", "Delete this saved run?") != QMessageBox.StandardButton.Yes:
                return
            try:
                delete_run(path)
            except OSError as exc:
                QMessageBox.warning(self, "Can't delete", str(exc))
            self._reload_history()

        def _export(self, result, kind):
            from gpu_bench.store import export

            if not result:
                QMessageBox.information(self, "Nothing to export", "Run a comparison or pick a saved run first.")
                return
            name = result.get("workload", {}).get("name", "run")
            suggested, flt = {
                "summary": (f"{name}-summary.csv", "CSV (*.csv)"),
                "jobs": (f"{name}-jobs.csv", "CSV (*.csv)"),
                "json": (f"{name}.json", "JSON (*.json)"),
            }[kind]
            path, _ = QFileDialog.getSaveFileName(self, "Export", str(Path.home() / suggested), flt)
            if not path:
                return
            try:
                export(result, path, "jobs" if kind == "jobs" else "summary")
            except OSError as exc:
                QMessageBox.warning(self, "Export failed", str(exc))
                return
            self.status.setText(f"Wrote {path}")

        # misc ---------------------------------------------------------

        def _log(self, text):
            self.log.appendPlainText(text)

        def _save_settings(self):
            from gpu_bench.store import save_settings

            try:
                save_settings(self.settings)
            except OSError:
                pass

        def closeEvent(self, event):
            if self.task is not None and self.task.isRunning():
                answer = QMessageBox.question(self, "Run in progress", "A run is still going. Stop it and quit?")
                if answer != QMessageBox.StandardButton.Yes:
                    event.ignore()
                    return
                self.task.cancel.set()
                self.task.wait(5000)
            for task in (self.probe_task,):
                if task is not None and task.isRunning():
                    task.wait(2000)
            event.accept()


def _plain_table(table):
    table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    table.verticalHeader().setVisible(False)
    table.horizontalHeader().setStretchLastSection(False)
    table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)


def _safe_json(path):
    from gpu_bench.compare import load_json

    try:
        return load_json(path)
    except ValueError:
        return None


def _model_label(item):
    data = item.get("data") or {}
    if "backend" in data and "runtime" not in data:
        return f"{item['name']}   (live only)"
    if "backend" in data:
        return f"{item['name']}   (real backend)"
    return item["name"]


def _model_tooltip(item):
    from gpu_bench.backends import describe

    data = item.get("data") or {}
    return f"{item.get('description', '')}\nLive runs use: {describe(data)}\n{item['path']}"


def _fill_plain(widget, items, tooltip=None, label=None):
    widget.clear()
    for item in items:
        row = QListWidgetItem(label(item) if label else item["name"])
        if tooltip:
            row.setToolTip(tooltip(item))
        widget.addItem(row)


def _fill_checks(widget, items, checked=None):
    widget.clear()
    for item in items:
        row = QListWidgetItem(item["name"])
        row.setFlags(row.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        on = not checked or item["name"] in checked
        row.setCheckState(Qt.CheckState.Checked if on else Qt.CheckState.Unchecked)
        row.setToolTip(item.get("description", ""))
        widget.addItem(row)


def _current(widget, items):
    row = widget.currentRow()
    if row < 0 or row >= len(items):
        return None
    return items[row]


def _checked(widget, items):
    chosen = []
    for index, item in enumerate(items):
        row = widget.item(index)
        if row is not None and row.checkState() == Qt.CheckState.Checked:
            chosen.append(item)
    return chosen


_USER_COLORS = ("#9ecae9", "#a6d8a0", "#d4b9da", "#c7c7c7", "#fdd0a2", "#b4d8d5")


def _job_color(job):
    """Retakes stand out in red; other jobs are colored by user (muted), or by id without users."""
    if job.get("kind") == "retake":
        return QColor("#d62728")
    if job.get("user"):
        total = sum(ord(char) for char in str(job["user"]))
        return QColor(_USER_COLORS[total % len(_USER_COLORS)]).darker(115)
    return _bar_color(job["id"])


def _bar_color(job_id):
    total = sum(ord(char) for char in str(job_id))
    return QColor(_BAR_COLORS[total % len(_BAR_COLORS)])


if __name__ == "__main__":
    raise SystemExit(main())
