"""Small desktop window. The command line does not import this module."""

import sys

try:
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QColor, QPainter
    from PySide6.QtWidgets import (
        QApplication,
        QHBoxLayout,
        QLabel,
        QListWidget,
        QListWidgetItem,
        QPushButton,
        QTableWidget,
        QTableWidgetItem,
        QVBoxLayout,
        QWidget,
    )
except ImportError:
    Qt = None


_BAR_COLORS = (
    QColor("#4c78a8") if Qt else None,
    QColor("#f58518") if Qt else None,
    QColor("#54a24b") if Qt else None,
    QColor("#e45756") if Qt else None,
    QColor("#72b7b2") if Qt else None,
    QColor("#b279a2") if Qt else None,
)


def main():
    if Qt is None:
        print(
            "error: the desktop window needs PySide6.\nInstall it with: pip install PySide6",
            file=sys.stderr,
        )
        return 1
    app = QApplication(sys.argv)
    window = Window()
    window.resize(960, 640)
    window.show()
    return app.exec()


if Qt is not None:

    class Timeline(QWidget):
        """One lane per scheduler. Each bar is a request on the GPU."""

        def __init__(self):
            super().__init__()
            self.results = []
            self.setMinimumHeight(80)

        def set_results(self, results):
            self.results = results
            lanes = max(1, len(results))
            self.setMinimumHeight(52 + lanes * 30)
            self.update()

        def paintEvent(self, _event):
            painter = QPainter(self)
            if not self.results:
                painter.drawText(
                    self.rect(),
                    Qt.AlignmentFlag.AlignCenter,
                    "Run a comparison to see the schedule.",
                )
                return

            metrics = painter.fontMetrics()
            label_w = max(metrics.horizontalAdvance(item["name"]) for item in self.results) + 16
            makespan = max(job["finish_ms"] for item in self.results for job in item["jobs"]) or 1
            plot_w = max(1, self.width() - label_w - 12)
            lane_h = 22
            for index, item in enumerate(self.results):
                y = 8 + index * 30
                painter.drawText(
                    4,
                    y,
                    label_w - 12,
                    lane_h,
                    Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                    item["name"],
                )
                for job in item["jobs"]:
                    x = label_w + plot_w * job["start_ms"] / makespan
                    width = max(1, plot_w * (job["finish_ms"] - job["start_ms"]) / makespan)
                    painter.fillRect(int(x), y, int(width), lane_h, _bar_color(job["id"]))
                    if width >= 48:
                        painter.drawText(
                            int(x) + 4,
                            y,
                            int(width) - 8,
                            lane_h,
                            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                            job["id"],
                        )
            end = f"{makespan:.0f} ms"
            painter.drawText(label_w, self.height() - 18, "0 ms")
            painter.drawText(self.width() - metrics.horizontalAdvance(end) - 8, self.height() - 18, end)


    class Window(QWidget):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("GPU inference scheduler")
            from gpu_bench.catalog import model_items, scheduler_items, workload_items

            self._models = model_items()
            self._workloads = workload_items()
            self._schedulers = scheduler_items()

            self.models = QListWidget()
            self.workloads = QListWidget()
            self.schedulers = QListWidget()
            self.models.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
            self.workloads.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
            _fill_plain(self.models, self._models)
            _fill_plain(self.workloads, self._workloads)
            _fill_checks(self.schedulers, self._schedulers)
            if self.workloads.count():
                self.workloads.setCurrentRow(0)
            paired = self._workloads[0].get("model") if self._workloads else ""
            model_row = next((index for index, item in enumerate(self._models) if item["name"] == paired), 0)
            if self.models.count():
                self.models.setCurrentRow(min(model_row, self.models.count() - 1))

            run = QPushButton("Run")
            run.clicked.connect(self.run_comparison)
            self.status = QLabel("Pick a model, a workload, and at least one scheduler.")
            self.status.setWordWrap(True)

            chooser = QVBoxLayout()
            chooser.addWidget(QLabel("Model"))
            chooser.addWidget(self.models, 1)
            chooser.addWidget(QLabel("Workload"))
            chooser.addWidget(self.workloads, 1)
            chooser.addWidget(QLabel("Schedulers"))
            chooser.addWidget(self.schedulers, 1)
            chooser.addWidget(run)
            chooser.addWidget(self.status)

            self.table = QTableWidget(0, 5)
            self.table.setHorizontalHeaderLabels(["scheduler", "mean_ms", "p95_ms", "wait_ms", "jobs/s"])
            self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
            self.table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
            self.table.verticalHeader().setVisible(False)
            self.table.horizontalHeader().setStretchLastSection(True)
            self.note = QLabel("")
            self.note.setWordWrap(True)
            self.timeline = Timeline()

            results = QVBoxLayout()
            results.addWidget(self.table, 1)
            results.addWidget(self.note)
            results.addWidget(self.timeline, 1)

            layout = QHBoxLayout(self)
            layout.addLayout(chooser, 1)
            layout.addLayout(results, 2)

        def run_comparison(self):
            from gpu_bench.compare import comparison

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
            try:
                data = comparison(
                    workload["path"],
                    model["path"],
                    [(item["name"], item["path"]) for item in chosen],
                )
            except (OSError, ValueError, RuntimeError, KeyError) as exc:
                self.status.setText(f"error: {exc}")
                return
            self._show(data)
            self.status.setText(f"Compared {data['workload']['name']} on {data['model']['name']}.")

        def _show(self, data):
            results = data["results"]
            self.table.setRowCount(len(results))
            for row, item in enumerate(results):
                summary = item["summary"]
                values = [
                    item["name"],
                    f"{summary['mean_latency_ms']:.1f}",
                    f"{summary['p95_latency_ms']:.1f}",
                    f"{summary['mean_wait_ms']:.1f}",
                    f"{summary['jobs_per_s']:.2f}",
                ]
                for column, value in enumerate(values):
                    cell = QTableWidgetItem(value)
                    if column:
                        cell.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                    self.table.setItem(row, column, cell)
            self.table.resizeColumnsToContents()
            self.note.setText(_lowest_note(results))
            self.timeline.set_results(results)


def _fill_plain(widget, items):
    widget.clear()
    for item in items:
        widget.addItem(item["name"])


def _fill_checks(widget, items):
    widget.clear()
    for item in items:
        row = QListWidgetItem(item["name"])
        row.setFlags(row.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        row.setCheckState(Qt.CheckState.Checked)
        widget.addItem(row)


def _current(widget, items):
    row = widget.currentRow()
    if row < 0 or row >= len(items):
        return None
    return items[row]


def _checked(widget, items):
    chosen = []
    for index, item in enumerate(items):
        if widget.item(index).checkState() == Qt.CheckState.Checked:
            chosen.append(item)
    return chosen


def _bar_color(job_id):
    total = sum(ord(char) for char in str(job_id))
    return _BAR_COLORS[total % len(_BAR_COLORS)]


def _lowest_note(results):
    if len(results) < 2:
        return ""
    return "\n".join(
        (
            _lowest(results, "mean_latency_ms", "mean latency"),
            _lowest(results, "p95_latency_ms", "p95 latency"),
        )
    )


def _lowest(results, key, label):
    scored = [(round(item["summary"][key], 1), item["name"]) for item in results]
    best = min(score for score, _name in scored)
    names = [name for score, name in scored if score == best]
    return f"lowest {label}: {', '.join(names)} ({best:.1f} ms)"


if __name__ == "__main__":
    raise SystemExit(main())
