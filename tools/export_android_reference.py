"""Export the Python detector's exact reading of the sample boards for the APK tests.

Run this whenever the shared vision/solver/counter logic changes::

    python tools/export_android_reference.py

It writes ``android/app/src/test/resources/reference.tsv``, which the JVM tests
compare against.  The APK cannot use Python, so this file is the bridge that
proves the Java port still reads real screenshots the same way.
"""

from __future__ import annotations

import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arrowbot import counter, solver, vision  # noqa: E402
from arrowbot.config import Config  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLES = ("real_level.png", "real_level_74.png", "real_level_zoomed.png")
OUT = os.path.join(ROOT, "android", "app", "src", "test", "resources", "reference.tsv")


def main() -> int:
    detector = vision.Detector(Config())
    lines = ["# arrowpuzzle android reference v1",
             "# file\tcount\tcounter\tplan\tscale\tbx0\tby0\tbx1\tby1"]
    arrows = ["# arrows: file\tx\ty\tdirection\tmargin"]
    for name in SAMPLES:
        path = os.path.join(ROOT, "samples", name)
        image = cv2.imread(path)
        if image is None:
            raise SystemExit(f"cannot read {path}")
        mask = detector.ink(image)
        # the APK uses an exact Euclidean distance transform, so the reference
        # must be made with OpenCV's precise transform too (not the 5x5 mask)
        distance = cv2.distanceTransform(mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        scale = vision.estimate_scale(mask, distance)
        detected = detector.arrows(mask, scale, distance)
        waves = solver.plan(detected)
        if waves is None:
            raise SystemExit(f"{name}: deadlocked")
        count = len(detected)
        value = counter.read_counter(image)
        bbox = vision.bbox_of(mask)
        lines.append("\t".join([
            name, str(count), str(value), str(sum(len(wave) for wave in waves)),
            f"{scale:.6f}", *[str(v) for v in bbox],
        ]))
        for arrow in sorted(detected, key=lambda item: (item.cy, item.cx)):
            arrows.append("\t".join([
                name, str(arrow.cx), str(arrow.cy), arrow.direction,
                f"{arrow.confidence:.6f}",
            ]))
        print(f"{name}: {count} arrows, counter={value}, "
              f"{len(waves)} waves, scale={scale:.4f}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines + arrows) + "\n")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
