"""Charts for the desktop window, painted with Qt (no plotting library needed).

LatencyBars: mean latency per scheduler as bars, p95 as a tick above each bar,
one bar per kind of request when the workload has kinds (retake, script_segment).
LatencyCDF: share of requests finished within a given latency, one line per scheduler.
"""

import math

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

SERIES_COLORS = ("#4c78a8", "#f58518", "#54a24b", "#e45756", "#72b7b2", "#b279a2", "#eeca3b", "#9d755d")
ALL = "all requests"


def nice_ticks(top, count=5):
    """Round tick values from 0 to at least top."""
    if top <= 0:
        return [0, 1]
    raw = top / count
    magnitude = 10 ** math.floor(math.log10(raw))
    step = next(m * magnitude for m in (1, 2, 2.5, 5, 10) if m * magnitude >= raw)
    ticks = [0.0]
    while ticks[-1] < top:  # the last tick must reach top, or the largest value is drawn off the axis
        ticks.append(round(ticks[-1] + step, 10))
    return ticks


def format_ms(value, use_seconds):
    if use_seconds:
        return f"{value / 1000:g} s" if value % 1000 == 0 else f"{value / 1000:.1f} s"
    return f"{value:g} ms"


def kinds_of(result):
    kinds = []
    for item in (result or {}).get("results", []):
        for kind in (item["summary"].get("by_kind") or {}):
            if kind not in kinds:
                kinds.append(kind)
    return kinds


class _Chart(QWidget):
    def __init__(self, title):
        super().__init__()
        self.title = title
        self.result = None
        self.setMinimumHeight(260)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_result(self, result):
        self.result = result
        self.update()

    def _begin(self):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), self.palette().color(self.backgroundRole()))
        self.text = self.palette().color(self.foregroundRole())
        self.grid = QColor(self.text)
        self.grid.setAlpha(40)
        return painter

    def _empty(self, painter, message="Run a comparison to see this chart."):
        painter.setPen(self.text)
        painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, message)

    def _title(self, painter, subtitle=""):
        font = QFont(painter.font())
        font.setBold(True)
        font.setPointSizeF(font.pointSizeF() * 1.15)
        painter.setFont(font)
        painter.setPen(self.text)
        painter.drawText(QRectF(12, 6, self.width() - 24, 24), Qt.AlignmentFlag.AlignLeft, self.title)
        painter.setFont(QFont(self.font()))
        if subtitle:
            dim = QColor(self.text)
            dim.setAlpha(150)
            painter.setPen(dim)
            painter.drawText(QRectF(12, 28, self.width() - 24, 18), Qt.AlignmentFlag.AlignLeft, subtitle)

    def _legend(self, painter, names, top):
        metrics = painter.fontMetrics()
        x = self.width() - 12
        for index, name in reversed(list(enumerate(names))):
            width = metrics.horizontalAdvance(name) + 22
            x -= width
            painter.fillRect(QRectF(x, top + 3, 12, 12), QColor(SERIES_COLORS[index % len(SERIES_COLORS)]))
            painter.setPen(self.text)
            painter.drawText(QRectF(x + 16, top, width, 18), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, name)

    def _y_axis(self, painter, plot, top_value):
        ticks = nice_ticks(top_value)
        use_seconds = ticks[-1] >= 5000
        metrics = painter.fontMetrics()
        for tick in ticks:
            y = plot.bottom() - plot.height() * tick / ticks[-1]
            painter.setPen(QPen(self.grid, 1))
            painter.drawLine(QPointF(plot.left(), y), QPointF(plot.right(), y))
            label = format_ms(tick, use_seconds)
            painter.setPen(self.text)
            painter.drawText(QRectF(0, y - 9, plot.left() - 8, 18),
                             Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, label)
        return ticks[-1], use_seconds, metrics


class LatencyBars(_Chart):
    def __init__(self):
        super().__init__("Latency by scheduler")

    def paintEvent(self, _event):
        painter = self._begin()
        rows = (self.result or {}).get("results") or []
        if not rows:
            self._empty(painter)
            return
        kinds = kinds_of(self.result) or [ALL]

        def stats(item, kind):
            summary = item["summary"]
            return summary if kind == ALL else (summary.get("by_kind") or {}).get(kind)

        self._title(painter, "bar = mean latency (arrival to finish), tick = p95")
        if len(kinds) > 1 or kinds[0] != ALL:
            self._legend(painter, kinds, 28)
        top = max(
            max(s["mean_latency_ms"], s["p95_latency_ms"])
            for item in rows for kind in kinds for s in [stats(item, kind)] if s
        )
        plot = QRectF(70, 56, self.width() - 86, self.height() - 56 - 44)
        top, use_seconds, metrics = self._y_axis(painter, plot, top * 1.08)

        group_w = plot.width() / len(rows)
        bar_w = min(56.0, group_w * 0.8 / len(kinds))
        for index, item in enumerate(rows):
            center = plot.left() + group_w * (index + 0.5)
            start = center - bar_w * len(kinds) / 2
            for k, kind in enumerate(kinds):
                s = stats(item, kind)
                if not s:
                    continue
                color = QColor(SERIES_COLORS[k % len(SERIES_COLORS)])
                x = start + k * bar_w
                height = plot.height() * s["mean_latency_ms"] / top
                bar = QRectF(x + 2, plot.bottom() - height, bar_w - 4, height)
                painter.fillRect(bar, color)
                p95_y = plot.bottom() - plot.height() * s["p95_latency_ms"] / top
                painter.setPen(QPen(color.darker(150), 2))
                painter.drawLine(QPointF(x + bar_w * 0.2, p95_y), QPointF(x + bar_w * 0.8, p95_y))
                painter.drawLine(QPointF(x + bar_w / 2, p95_y), QPointF(x + bar_w / 2, bar.top()))
                if bar_w >= 34:
                    label = format_ms(round(s["mean_latency_ms"], -1 if not use_seconds else -2), use_seconds)
                    if bar.height() >= 22:  # inside the bar, clear of the p95 line
                        painter.setPen(QColor("white"))
                        painter.drawText(QRectF(x, bar.top() + 2, bar_w, 16), Qt.AlignmentFlag.AlignCenter, label)
                    else:
                        painter.setPen(self.text)
                        painter.drawText(QRectF(x - 10, min(bar.top(), p95_y) - 18, bar_w + 20, 16),
                                         Qt.AlignmentFlag.AlignCenter, label)
            painter.setPen(self.text)
            name = metrics.elidedText(item["name"], Qt.TextElideMode.ElideRight, int(group_w - 6))
            painter.drawText(QRectF(center - group_w / 2, plot.bottom() + 6, group_w, 20), Qt.AlignmentFlag.AlignCenter, name)


class LatencyCDF(_Chart):
    def __init__(self):
        super().__init__("Share of requests finished within a latency")
        self.kind = ALL

    def set_kind(self, kind):
        self.kind = kind or ALL
        self.update()

    def _latencies(self, item):
        jobs = item.get("jobs") or []
        if self.kind != ALL:
            jobs = [job for job in jobs if job.get("kind") == self.kind]
        return sorted(job["latency_ms"] for job in jobs)

    def paintEvent(self, _event):
        painter = self._begin()
        rows = (self.result or {}).get("results") or []
        series = [(item["name"], self._latencies(item)) for item in rows]
        series = [(name, values) for name, values in series if values]
        if not series:
            self._empty(painter)
            return
        self._title(painter, f"{self.kind}: further left is better")
        self._legend(painter, [name for name, _values in series], 28)
        top = max(values[-1] for _name, values in series) * 1.02 or 1
        plot = QRectF(56, 56, self.width() - 72, self.height() - 56 - 44)
        ticks = nice_ticks(top)
        use_seconds = ticks[-1] >= 5000
        for fraction in (0, 0.25, 0.5, 0.75, 1.0):
            y = plot.bottom() - plot.height() * fraction
            painter.setPen(QPen(self.grid, 1))
            painter.drawLine(QPointF(plot.left(), y), QPointF(plot.right(), y))
            painter.setPen(self.text)
            painter.drawText(QRectF(0, y - 9, plot.left() - 8, 18),
                             Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, f"{fraction:.0%}")
        for tick in ticks:
            x = plot.left() + plot.width() * tick / ticks[-1]
            painter.setPen(QPen(self.grid, 1))
            painter.drawLine(QPointF(x, plot.top()), QPointF(x, plot.bottom()))
            painter.setPen(self.text)
            painter.drawText(QRectF(x - 40, plot.bottom() + 6, 80, 18), Qt.AlignmentFlag.AlignCenter,
                             format_ms(tick, use_seconds))
        for index, (_name, values) in enumerate(series):
            path = QPainterPath(QPointF(plot.left(), plot.bottom()))
            count = len(values)
            for rank, value in enumerate(values, start=1):
                x = plot.left() + plot.width() * value / ticks[-1]
                path.lineTo(QPointF(x, plot.bottom() - plot.height() * (rank - 1) / count))
                path.lineTo(QPointF(x, plot.bottom() - plot.height() * rank / count))
            painter.setPen(QPen(QColor(SERIES_COLORS[index % len(SERIES_COLORS)]), 2.2))
            painter.drawPath(path)
