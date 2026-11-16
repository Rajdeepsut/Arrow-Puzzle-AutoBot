"""Configuration for the Arrow Puzzle agent (JSON backed dataclass)."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, List, Optional

DEFAULT_CONFIG_PATH = "config.json"


@dataclass
class Config:
    # ------------------------------------------------------------------ device
    adb_path: str = "adb"
    serial: Optional[str] = None

    # ------------------------------------------------------------------ vision
    #: region of the screenshot that may contain the board, as fractions
    #: (x0, y0, x1, y1).  Everything outside is ignored.
    roi: List[float] = field(default_factory=lambda: [0.0, 0.10, 1.0, 0.90])
    #: extra ignored rectangles, same fraction format (ads, banners, buttons)
    exclude_rects: List[List[float]] = field(default_factory=list)
    #: plausible glyph size, as a fraction of the full screen area.  The upper
    #: bound only has to reject the board panel / big UI blocks, so it is
    #: deliberately generous: a 4x4 board has huge cells and therefore huge
    #: arrows.
    min_blob_area_frac: float = 0.00004
    max_blob_area_frac: float = 0.06
    #: odd kernel used to close small gaps inside a glyph (0 disables)
    morph_close: int = 3
    #: how far a blob may sit from the fitted lattice, relative to cell size
    spacing_tolerance: float = 0.22
    #: arrows classified below this confidence are never tapped (they still
    #: block rays, because a glyph we cannot read is still physically there)
    min_orientation_confidence: float = 0.60

    # ----------------------------------------------------------------- tapping
    tap_delay_ms: int = 140
    #: taps performed between two screen reads inside one wave.  3 is a good
    #: safety/speed trade-off; 1 is the most conservative.
    chunk_size: int = 3
    settle_timeout_ms: int = 3500
    settle_poll_ms: int = 120
    #: mean absolute grey difference (0-255) below which two frames count as
    #: "the animation finished"
    settle_thresh: float = 2.0
    #: how often a wave without any visible effect is retried before giving up
    tap_retries: int = 2

    # -------------------------------------------------------------- level flow
    max_waves_per_level: int = 400
    #: attempts to get past the level-complete screen
    advance_attempts: int = 6
    advance_wait_ms: int = 900
    #: fallback tap targets for the "next level" button, as screen fractions
    next_tap_points: List[List[float]] = field(
        default_factory=lambda: [[0.5, 0.86], [0.5, 0.80], [0.5, 0.91]]
    )
    #: "stop" (report and end) or "tap_anyway" (sacrifice a life and continue)
    on_deadlock: str = "stop"
    #: keep going when a level does not finish cleanly
    continue_on_failure: bool = False

    # --------------------------------------------------------------------- run
    max_levels: int = 500
    #: only used for log labels; the agent always plays whatever is on screen
    start_level: int = 1
    stop_on_error: bool = False
    write_debug: bool = True
    debug_dir: str = "debug"

    # ------------------------------------------------------------------ helpers
    @classmethod
    def load(cls, path: Optional[str] = None) -> "Config":
        cfg = cls()
        if path is None and os.path.exists(DEFAULT_CONFIG_PATH):
            path = DEFAULT_CONFIG_PATH
        if path:
            with open(path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            cfg.update(data)
        return cfg.validate()

    def update(self, data: Dict[str, Any]) -> "Config":
        known = {f.name for f in fields(self)}
        for key, value in data.items():
            if key in known:
                setattr(self, key, value)
        return self

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(asdict(self), handle, indent=2)
            handle.write("\n")

    def validate(self) -> "Config":
        roi = list(self.roi or [0.0, 0.0, 1.0, 1.0])
        if len(roi) != 4:
            roi = [0.0, 0.0, 1.0, 1.0]
        x0, y0, x1, y1 = (min(1.0, max(0.0, float(v))) for v in roi)
        if x1 - x0 < 0.05:
            x0, x1 = 0.0, 1.0
        if y1 - y0 < 0.05:
            y0, y1 = 0.0, 1.0
        self.roi = [x0, y0, x1, y1]

        rects: List[List[float]] = []
        for rect in self.exclude_rects or []:
            if len(rect) == 4:
                rects.append([float(min(1.0, max(0.0, v))) for v in rect])
        self.exclude_rects = rects

        if self.on_deadlock not in ("stop", "tap_anyway"):
            self.on_deadlock = "stop"

        self.min_blob_area_frac = max(1e-6, float(self.min_blob_area_frac))
        self.max_blob_area_frac = max(
            self.min_blob_area_frac * 4.0, float(self.max_blob_area_frac)
        )
        self.morph_close = int(self.morph_close)
        if self.morph_close and self.morph_close < 3:
            self.morph_close = 3
        if self.morph_close % 2 == 0:
            self.morph_close += 1

        self.chunk_size = max(1, int(self.chunk_size))
        self.max_levels = max(1, int(self.max_levels))
        self.max_waves_per_level = max(1, int(self.max_waves_per_level))
        self.tap_retries = max(0, int(self.tap_retries))
        self.min_orientation_confidence = float(
            min(1.0, max(0.0, self.min_orientation_confidence))
        )

        points: List[List[float]] = []
        for point in self.next_tap_points or []:
            if len(point) == 2:
                points.append([float(min(1.0, max(0.0, point[0]))),
                               float(min(1.0, max(0.0, point[1])))])
        self.next_tap_points = points or [[0.5, 0.86]]
        return self
