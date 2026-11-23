"""Render the app icon and the README images.

    python docs/make_images.py                      # simulated narration-small run
    python docs/make_images.py path/to/result.json  # a saved (live) result, e.g. from `history export`

Writes gpu_bench/assets/icon.png, gpu_bench/assets/icon.ico, docs/images/charts.png, docs/images/timeline.png.
"""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
if sys.platform == "win32":
    os.environ.setdefault("QT_QPA_FONTDIR", "C:/Windows/Fonts")

from PySide6.QtCore import QRectF, Qt  # noqa: E402
from PySide6.QtGui import QColor, QImage, QPainter, QPainterPath, QPalette  # noqa: E402
from PySide6.QtWidgets import QApplication, QVBoxLayout, QWidget  # noqa: E402


def draw_icon(size):
    image = QImage(size, size, QImage.Format.Format_ARGB32)
    image.fill(Qt.GlobalColor.transparent)
    p = QPainter(image)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    path = QPainterPath()
    path.addRoundedRect(QRectF(0, 0, size, size), size * 0.2, size * 0.2)
    p.fillPath(path, QColor("#1f2937"))
    # three lanes of jobs; the red one jumps the queue
    lane_h, gap = size * 0.16, size * 0.08
    top = (size - 3 * lane_h - 2 * gap) / 2
    lanes = [
        [(0.12, 0.30, "#4c78a8"), (0.33, 0.62, "#4c78a8"), (0.65, 0.88, "#4c78a8")],
        [(0.12, 0.22, "#e45756"), (0.25, 0.55, "#54a24b"), (0.58, 0.88, "#54a24b")],
        [(0.12, 0.45, "#f58518"), (0.48, 0.70, "#f58518"), (0.73, 0.88, "#f58518")],
    ]
    for index, lane in enumerate(lanes):
        y = top + index * (lane_h + gap)
        for start, end, color in lane:
            bar = QPainterPath()
            bar.addRoundedRect(QRectF(size * start, y, size * (end - start), lane_h), lane_h * 0.3, lane_h * 0.3)
            p.fillPath(bar, QColor(color))
    p.end()
    return image


def main():
    app = QApplication([])
    from PySide6.QtGui import QFont, QFontDatabase

    for family in ("Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"):
        if family in QFontDatabase.families():
            app.setFont(QFont(family, 10))
            break
    palette = app.palette()
    palette.setColor(QPalette.ColorRole.Window, QColor("white"))
    palette.setColor(QPalette.ColorRole.WindowText, QColor("#111827"))
    app.setPalette(palette)

    assets = ROOT / "gpu_bench" / "assets"
    assets.mkdir(exist_ok=True)
    draw_icon(256).save(str(assets / "icon.png"))
    if not draw_icon(256).save(str(assets / "icon.ico"), "ICO"):
        print("could not write icon.ico (Qt ICO plugin missing); build_exe.py will skip the icon")

    from gpu_bench.charts import LatencyBars, LatencyCDF
    from gpu_bench.compare import MODELS, SCHEDULERS, WORKLOADS
    from gpu_bench.gui import Timeline
    from gpu_bench.runner import execute

    if len(sys.argv) > 1:
        result = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    else:
        names = ("fifo", "shortest_predicted", "fair_share", "retake_first")
        result = execute(WORKLOADS / "narration-small.json", MODELS / "example-tts.json",
                         [(name, SCHEDULERS / f"{name}.py") for name in names])

    images = ROOT / "docs" / "images"
    images.mkdir(parents=True, exist_ok=True)
    box = QWidget()
    box.setAutoFillBackground(True)
    layout = QVBoxLayout(box)
    bars, cdf = LatencyBars(), LatencyCDF()
    cdf.set_kind("retake" if any("by_kind" in item["summary"] for item in result["results"]) else "all requests")
    bars.set_result(result)
    cdf.set_result(result)
    layout.addWidget(bars)
    layout.addWidget(cdf)
    box.resize(1100, 760)
    box.show()
    app.processEvents()
    box.grab().save(str(images / "charts.png"))

    rows = [item for item in result["results"] if item["name"] in ("fifo", "fair_share")] or result["results"]
    timeline = Timeline()
    timeline.setAutoFillBackground(True)
    timeline.set_results(rows)
    timeline.resize(1100, 56 + 30 * len(rows))
    timeline.show()
    app.processEvents()
    timeline.grab().save(str(images / "timeline.png"))
    print(f"wrote {assets / 'icon.png'}, {images / 'charts.png'}, {images / 'timeline.png'}")


if __name__ == "__main__":
    main()
