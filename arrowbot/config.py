"""Configuration for the Arrow Puzzle agent (JSON backed dataclass).

Every knob has a default that was measured on a real 1080x2392 phone, so the bot
works without a config file.  :meth:`Config.load` reads ``config.json`` when it
exists and ignores unknown keys, which keeps old config files usable.
"""

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
    #: ``scrcpy`` (fast, true multi-touch) or ``adb`` (one-finger fallback)
    input_backend: str = "scrcpy"
    #: optional path to the scrcpy server jar (downloaded and verified if absent)
    scrcpy_jar: Optional[str] = None
    scrcpy_port: int = 27183
    scrcpy_log_level: str = "error"

    # ------------------------------------------------------------------ vision
    #: region that may contain the board, as screen fractions
    #: ``(x0, y0, x1, y1)``.  The default hides the HUD on top and the buttons
    #: at the bottom of a 1080x2392 phone.
    roi: List[float] = field(default_factory=lambda: [0.0, 0.14, 1.0, 0.79])
    #: extra ignored rectangles, same fraction format (ads, banners, popups)
    exclude_rects: List[List[float]] = field(default_factory=list)
    #: grey value below which a pixel counts as navy arrow ink
    ink_gray_max: int = 110
    #: red arrows are real arrows too and must not be missed
    red_ink: bool = True
    #: the camera zooms as a board empties; this is the accepted artwork scale
    scale_range: List[float] = field(default_factory=lambda: [0.5, 4.0])
    #: minimum template margin before an arrow may be tapped (0..1).  Real
    #: boards score 0.15-0.30 (median 0.25); 0.06 only rejects genuine ambiguity.
    min_direction_margin: float = 0.06
    #: an arrow smaller than this many pixels (at scale 1) is noise
    min_arrow_pixels: int = 60
    #: optional fixed minimum head size, in pixels of the distance transform
    min_head_pixels: Optional[int] = None

    # ----------------------------------------------------------------- tapping
    #: plan the whole board once, then tap a batch at a time (much faster)
    fast_mode: bool = True
    #: arrows per batch before the board is read and checked again
    batch_size: int = 8
    #: first batch of a level; grows towards ``batch_size`` while the camera is still
    batch_size_start: int = 6
    #: a surviving arrow moving more than this between batches means the camera moved
    batch_drift_px: int = 3
    #: let the batch grow back to ``batch_size`` when the camera stays put
    adaptive_batches: bool = True
    #: pause between two taps of one batch (scrcpy only needs a few ms)
    batch_gap_ms: int = 6
    #: how long each finger stays down for one arrow
    tap_hold_ms: int = 12
    #: pause after a batch, to let the removal animation finish
    batch_wait_ms: int = 160
    #: taps before re-reading the board in slow mode.  The camera can move as
    #: soon as an arrow disappears, so slow mode never batches.
    chunk_size: int = 1
    #: pause between taps of one batch in slow mode
    tap_delay_ms: int = 0
    #: pause after a tap, before the board is read again
    tap_wait_ms: int = 700
    #: screenshot interval while waiting for the camera to stop moving
    settle_poll_ms: int = 140
    #: give up waiting for a stable board after this long
    settle_timeout_ms: int = 5000
    #: two frames count as "the same view" when every head moved less than this
    stability_tol_px: int = 2
    #: check the tapped arrow is gone and no heart was lost
    verify_taps: bool = True
    #: how often a tap without any visible effect is retried before giving up
    tap_retries: int = 2
    #: never tap closer than this to the screen edge (clipped ink misleads)
    edge_margin_px: int = 18
    #: strip that shows the hearts, as screen fractions (kept strictly above
    #: ``roi`` so the red hearts are never mistaken for red arrows)
    hearts_roi: List[float] = field(default_factory=lambda: [0.31, 0.095, 0.72, 0.132])
    #: fewer red pixels than this means the HUD is hidden (banner) - skip the check
    hearts_min_red: int = 300

    # -------------------------------------------------------------------- zoom
    #: step 1 of every level: make sure the whole board is on screen
    auto_zoom: bool = True
    #: two-finger gesture centre, as screen fractions
    zoom_center: List[float] = field(default_factory=lambda: [0.5, 0.45])
    #: finger separation at the start and at the end of a pinch, in pixels
    zoom_start_gap_px: int = 280
    zoom_end_gap_px: int = 70
    zoom_steps: int = 8
    zoom_duration_ms: int = 180
    #: how many pinch gestures may be spent on one level
    zoom_attempts: int = 3
    #: wait after a pinch before the camera is read again
    zoom_settle_ms: int = 450
    #: ink this close to the edge of the board region counts as clipped
    zoom_edge_px: int = 3

    # -------------------------------------------------------------- level flow
    #: hard stop for one level (safety valve, not a normal limit)
    max_waves_per_level: int = 400
    #: how long to wait for the next board after a level is cleared
    advance_timeout_ms: int = 20000
    #: fallback taps allowed if the game shows a button instead of auto-advancing
    advance_attempts: int = 6
    advance_wait_ms: int = 900
    #: fallback "next level" button positions, as screen fractions
    next_tap_points: List[List[float]] = field(
        default_factory=lambda: [[0.5, 0.86], [0.5, 0.80], [0.5, 0.91]]
    )
    #: ``stop`` or ``tap_anyway`` (sacrifice a heart and keep playing)
    on_deadlock: str = "stop"
    #: keep going after a level that did not finish cleanly
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
        self.roi = _fraction_box(self.roi, [0.0, 0.0, 1.0, 1.0])
        self.hearts_roi = _fraction_box(self.hearts_roi, [0.0, 0.0, 1.0, 1.0])
        self.exclude_rects = [
            _fraction_box(rect, None) for rect in self.exclude_rects or [] if len(rect) == 4
        ]

        low, high = (float(v) for v in (self.scale_range or [0.5, 4.0])[:2])
        low = max(0.1, low)
        self.scale_range = [low, max(low * 1.5, high)]

        self.ink_gray_max = int(min(250, max(1, self.ink_gray_max)))
        self.min_direction_margin = float(min(0.9, max(0.0, self.min_direction_margin)))
        self.min_arrow_pixels = max(8, int(self.min_arrow_pixels))
        if self.min_head_pixels is not None:
            self.min_head_pixels = max(1, int(self.min_head_pixels))

        self.chunk_size = max(1, int(self.chunk_size))
        self.tap_delay_ms = max(0, int(self.tap_delay_ms))
        self.tap_wait_ms = max(0, int(self.tap_wait_ms))
        self.batch_size = max(1, int(self.batch_size))
        self.batch_size_start = max(1, min(self.batch_size, int(self.batch_size_start)))
        self.batch_drift_px = max(0, int(self.batch_drift_px))
        self.batch_gap_ms = max(0, int(self.batch_gap_ms))
        self.tap_hold_ms = max(0, int(self.tap_hold_ms))
        self.batch_wait_ms = max(0, int(self.batch_wait_ms))
        self.settle_poll_ms = max(10, int(self.settle_poll_ms))
        self.settle_timeout_ms = max(self.settle_poll_ms * 2, int(self.settle_timeout_ms))
        self.stability_tol_px = max(0, int(self.stability_tol_px))
        self.tap_retries = max(0, int(self.tap_retries))
        self.edge_margin_px = max(0, int(self.edge_margin_px))
        self.hearts_min_red = max(0, int(self.hearts_min_red))

        center = list(self.zoom_center or [0.5, 0.45])
        if len(center) != 2:
            center = [0.5, 0.45]
        self.zoom_center = [min(1.0, max(0.0, float(center[0]))),
                             min(1.0, max(0.0, float(center[1])))]
        self.zoom_start_gap_px = max(20, int(self.zoom_start_gap_px))
        self.zoom_end_gap_px = max(10, int(self.zoom_end_gap_px))
        self.zoom_steps = max(1, int(self.zoom_steps))
        self.zoom_duration_ms = max(0, int(self.zoom_duration_ms))
        self.zoom_attempts = max(0, int(self.zoom_attempts))
        self.zoom_settle_ms = max(0, int(self.zoom_settle_ms))
        self.zoom_edge_px = max(0, int(self.zoom_edge_px))
        if self.input_backend not in ("scrcpy", "adb"):
            self.input_backend = "scrcpy"

        self.max_levels = max(1, int(self.max_levels))
        self.max_waves_per_level = max(1, int(self.max_waves_per_level))
        self.advance_timeout_ms = max(100, int(self.advance_timeout_ms))
        self.advance_attempts = max(0, int(self.advance_attempts))
        self.advance_wait_ms = max(0, int(self.advance_wait_ms))
        if self.on_deadlock not in ("stop", "tap_anyway"):
            self.on_deadlock = "stop"

        points: List[List[float]] = []
        for point in self.next_tap_points or []:
            if len(point) == 2:
                points.append([float(min(1.0, max(0.0, point[0]))),
                               float(min(1.0, max(0.0, point[1])))])
        self.next_tap_points = points or [[0.5, 0.86]]
        return self


def _fraction_box(values, fallback: Optional[List[float]]) -> List[float]:
    """Clamp a ``(x0, y0, x1, y1)`` fraction box to something usable."""
    if not values or len(values) != 4:
        return list(fallback) if fallback else []
    x0, y0, x1, y1 = (min(1.0, max(0.0, float(v))) for v in values)
    if x1 - x0 < 0.01:
        x0, x1 = 0.0, 1.0
    if y1 - y0 < 0.01:
        y0, y1 = 0.0, 1.0
    return [x0, y0, x1, y1]
