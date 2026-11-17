"""The level loop: read the board, tap safe arrows, verify, repeat.

Two modes share the same safety rule (*an arrow may only be tapped when nothing
is in its corridor*):

* **fast** (default) - the whole board is planned once, then tapped in batches
  over scrcpy.  Removing an arrow can never block another one, so every arrow in
  that plan is safe in that exact order; a screenshot between batches catches a
  panned camera and lost hearts.  At the default view the board does not move at
  all between taps, which makes batching safe (measured: 0.0 px on 73 heads).
* **slow** - one tap per screenshot, for a zoomed-in or unstable camera.

Before any tap the camera is zoomed out (two-finger pinch) until no ink touches
the edge of the board region, so the detector sees the complete board.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import List, Optional, Protocol, Sequence, Tuple

import cv2
import numpy as np

from . import solver
from .config import Config
from .vision import Arrow, Detection, Detector, fractions_to_box, render_overlay


class Device(Protocol):
    """Anything that can show a screen and accept taps (adb or the simulator)."""

    def screencap(self) -> np.ndarray: ...

    def tap(self, x: int, y: int) -> None: ...


#: what to tell the user when a level does not finish cleanly
_HINTS = {
    "empty": "no arrows on screen - open the game on a level, or widen 'roi'",
    "unstable": "the camera never stopped moving - a popup or an ad is animating",
    "changed": "something new appeared on the board - a popup or ad is covering it",
    "stalled": "taps had no visible effect - a dialog is probably waiting",
    "deadlock": "no arrow can move (misread arrow, or a special tile) - see debug image",
    "life_lost": "a heart was lost (a blocked arrow was tapped) - stopped to be safe",
    "exhausted": "gave up after 'max_waves_per_level'",
    "error": "unexpected error - see the traceback above",
}


@dataclass
class Stats:
    levels_cleared: int = 0
    failed_levels: int = 0
    taps: int = 0
    deadlocks: int = 0
    hearts_lost: int = 0
    errors: int = 0
    level_times: List[float] = field(default_factory=list)
    started: float = field(default_factory=time.time)

    @property
    def elapsed(self) -> float:
        return time.time() - self.started

    @property
    def fastest(self) -> float:
        return min(self.level_times) if self.level_times else 0.0

    @property
    def average(self) -> float:
        return sum(self.level_times) / len(self.level_times) if self.level_times else 0.0

    def summary(self) -> str:
        minutes = self.elapsed / 60.0
        rate = self.levels_cleared / minutes if minutes > 0 else 0.0
        speed = ""
        if self.level_times:
            speed = (f" | fastest level: {self.fastest:.1f} s, "
                     f"average: {self.average:.1f} s")
        return (
            f"levels cleared: {self.levels_cleared} | failed: {self.failed_levels} | "
            f"taps: {self.taps} | deadlocks: {self.deadlocks} | "
            f"hearts lost: {self.hearts_lost} | errors: {self.errors} | "
            f"time: {minutes:.1f} min ({rate:.1f} levels/min){speed}"
        )


class ArrowAgent:
    """Plays Arrow Puzzle until the configured level budget is exhausted."""

    def __init__(
        self,
        config: Config,
        device: Device,
        detector: Optional[Detector] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.config = config
        self.device = device
        self.detector = detector or Detector(config)
        self.log = logger or logging.getLogger("arrowbot")
        self.stats = Stats()
        self._sleep = time.sleep
        self._hearts_seen = 0
        self._frame_shape: Optional[Tuple[int, int]] = None
        #: which pinch direction worked last time ("out" = fingers together)
        self._zoom_direction = "out"
        self._level_start_taps = 0

    # ------------------------------------------------------------------ input
    def read_stable(self) -> Optional[Detection]:
        """Capture until two consecutive frames show the same, unmoved board.

        Returns ``None`` when the view keeps moving for ``settle_timeout_ms``.
        """
        tolerance = self.config.stability_tol_px
        timeout = self.config.settle_timeout_ms / 1000.0
        poll = self.config.settle_poll_ms / 1000.0
        deadline = time.monotonic() + timeout
        previous: Optional[Detection] = None
        last: Optional[Detection] = None
        while time.monotonic() < deadline:
            image = self.device.screencap()
            self._frame_shape = image.shape[:2]
            detection = self.detector.detect(image)
            last = detection
            if previous is not None and detection.same_view(previous, tolerance):
                return detection
            previous = detection
            self._sleep(poll)
        if last is not None:
            self.log.debug("view never settled (%s)", last.describe())
        return None

    def tap(self, arrow: Arrow) -> None:
        x, y = arrow.tap_point()
        self.device.tap(x, y)
        self.stats.taps += 1
        if self.config.tap_delay_ms > 0:
            self._sleep(self.config.tap_delay_ms / 1000.0)

    def tap_batch(self, arrows: Sequence[Arrow]) -> int:
        """Tap several arrows as fast as the input backend allows."""
        arrows = list(arrows)
        points = [arrow.tap_point() for arrow in arrows]
        many = getattr(self.device, "tap_many", None)
        if callable(many):
            many(points, gap_ms=self.config.batch_gap_ms,
                 hold_ms=self.config.tap_hold_ms)
        else:
            gap = self.config.batch_gap_ms / 1000.0
            for x, y in points:
                self.device.tap(x, y)
                if gap:
                    self._sleep(gap)
        self.stats.taps += len(points)
        return len(points)

    def detect_once(self, retries: int = 2) -> Optional[Detection]:
        """One screenshot + detection, without waiting for the view to settle."""
        for attempt in range(max(1, retries)):
            try:
                image = self.device.screencap()
            except Exception as exc:  # a single dropped frame is not fatal
                self.log.debug("screencap failed (%d/%d): %s", attempt + 1, retries, exc)
                self._sleep(0.15)
                continue
            self._frame_shape = image.shape[:2]
            return self.detector.detect(image)
        return None

    # ------------------------------------------------------------------ zoom
    def board_clipped(self, detection: Detection) -> bool:
        """True when arrow ink runs into the edge of the board region."""
        if not detection.arrows:
            return False
        height, width = detection.image.shape[:2]
        rx0, ry0, rx1, ry1 = fractions_to_box(self.config.roi, width, height)
        margin = self.config.zoom_edge_px
        x0, y0, x1, y1 = detection.bbox
        return (x0 <= rx0 + margin or y0 <= ry0 + margin
                or x1 >= rx1 - 1 - margin or y1 >= ry1 - 1 - margin)

    def _fit_rank(self, detection: Detection) -> Tuple[int, int, float]:
        """Bigger is better: whole board first, then more arrows, then smaller ink."""
        return (0 if self.board_clipped(detection) else 1,
                len(detection.arrows), -detection.scale)

    def camera_drift(self, before: Detection, after: Detection) -> Optional[float]:
        """How far the surviving arrow heads moved between two detections.

        The game pans and zooms as a board empties; a batch is only safe while
        the artwork stands still.  Returns ``None`` when there are too few
        survivors to judge (for example after the final batch).
        """
        if not before.arrows or not after.arrows:
            return None
        radius = 24.0 * max(0.5, after.scale)
        used = set()
        distances = []
        for old in before.arrows:
            best_index = None
            best = radius
            for index, new in enumerate(after.arrows):
                if index in used:
                    continue
                distance = float(np.hypot(old.cx - new.cx, old.cy - new.cy))
                if distance < best:
                    best, best_index = distance, index
            if best_index is not None:
                used.add(best_index)
                distances.append(best)
        if len(distances) < 3:
            return None
        return max(distances)

    def _pinch(self, inward: bool) -> bool:
        pinch = getattr(self.device, "pinch", None)
        if not callable(pinch):
            return False
        height, width = self._frame_shape or (2392, 1080)
        centre_x = self.config.zoom_center[0] * width
        centre_y = self.config.zoom_center[1] * height
        if inward:
            start, end = self.config.zoom_start_gap_px, self.config.zoom_end_gap_px
        else:
            start, end = self.config.zoom_end_gap_px, self.config.zoom_start_gap_px
        self.log.info("pinching %s: %.0f -> %.0f px at (%.0f, %.0f)",
                      "together (zoom out)" if inward else "apart",
                      start, end, centre_x, centre_y)
        pinch(centre_x, centre_y, start, end,
              steps=self.config.zoom_steps,
              duration_ms=self.config.zoom_duration_ms)
        self._sleep(self.config.zoom_settle_ms / 1000.0)
        return True

    def ensure_zoom(self, detection: Optional[Detection] = None) -> Optional[Detection]:
        """Step 1 of a level: pinch until no arrow ink touches the board edge.

        The game remembers its camera zoom across levels, so a board can open
        partially off-screen.  Both pinch directions are tried and the one that
        actually shrinks the artwork is remembered.  When the view already fits
        (the normal case) nothing is sent.
        """
        if detection is None:
            detection = self.read_stable()
        if detection is None:
            return None
        if not self.config.auto_zoom:
            return detection

        if not self.board_clipped(detection):
            self.log.debug("view already fits (scale %.2f, bbox %s)",
                           detection.scale, detection.bbox)
            return detection
        if not callable(getattr(self.device, "pinch", None)):
            self.log.warning("board ink touches the edge, but this backend cannot "
                             "pinch - continuing with a partial board")
            return detection

        self.log.info("board is clipped at the edge -> zooming out (scale %.2f, %d arrows)",
                      detection.scale, len(detection.arrows))
        latest = detection
        direction = self._zoom_direction
        opposite = "in" if direction == "out" else "out"
        switched = False
        for _ in range(max(1, self.config.zoom_attempts)):
            if not self.board_clipped(latest):
                break
            if not self._pinch(direction == "out"):
                break
            after = self.read_stable()
            if after is None:
                continue
            improved = self._fit_rank(after) > self._fit_rank(latest)
            latest = after
            if improved:
                self._zoom_direction = direction
                continue
            if not switched:
                switched = True
                direction = opposite
                continue
            break

        if self.board_clipped(latest):
            self.log.warning("could not fit the whole board (scale %.2f, %d arrows)",
                             latest.scale, len(latest.arrows))
        else:
            self.log.info("zoomed out: scale %.2f, whole board visible (%d arrows)",
                          latest.scale, len(latest.arrows))
        return latest

    # ----------------------------------------------------------------- hearts
    def red_pixels(self, image: np.ndarray) -> int:
        height, width = image.shape[:2]
        x0, y0, x1, y1 = fractions_to_box(self.config.hearts_roi, width, height)
        if x1 <= x0 or y1 <= y0:
            return 0
        blue, green, red = cv2.split(image[y0:y1, x0:x1].astype(np.int16))
        return int(((red > 140) & (green < 110) & (blue < 110)).sum())

    def check_hearts(self, image: np.ndarray) -> Tuple[bool, int]:
        """``(no heart was lost, red pixels)``; a hidden HUD is not a failure."""
        red = self.red_pixels(image)
        if red < self.config.hearts_min_red:
            return True, red           # banner over the HUD - cannot tell, do not fail
        if self._hearts_seen and red < 0.85 * self._hearts_seen:
            self._hearts_seen = red
            self.stats.hearts_lost += 1
            return False, red
        self._hearts_seen = max(self._hearts_seen, red)
        return True, red

    # -------------------------------------------------------------- the board
    def candidates(self, detection: Detection) -> List[Arrow]:
        """Safe arrows that the bot is confident about, best first."""
        if not detection.arrows:
            return []
        edge = self.config.edge_margin_px
        width = detection.image.shape[1]
        confident = [arrow for arrow in detection.arrows
                     if arrow.confidence >= self.config.min_direction_margin]
        near_edge = [arrow for arrow in confident if not (edge <= arrow.cx < width - edge)]
        pool = [arrow for arrow in confident if arrow not in near_edge] or near_edge
        safe = [arrow for arrow in pool if solver.is_safe(arrow, detection.arrows)]
        return sorted(safe, key=lambda arrow: arrow.confidence, reverse=True)

    @staticmethod
    def looks_like_board(detection: Optional[Detection], minimum: int = 4) -> bool:
        return detection is not None and len(detection.arrows) >= minimum

    # ------------------------------------------------------------------ level
    def play_level(self, level: int = 0, dry_run: bool = False) -> str:
        """Clear one board.  Returns ``cleared``, ``dryrun`` or a failure reason."""
        if self.config.fast_mode:
            return self.play_level_fast(level, dry_run=dry_run)
        return self.play_level_slow(level, dry_run=dry_run)

    def all_tappable(self, detection: Detection) -> bool:
        """Fast mode taps a whole plan at once, so every arrow must be certain."""
        edge = self.config.edge_margin_px
        width = detection.image.shape[1]
        for arrow in detection.arrows:
            if arrow.confidence < self.config.min_direction_margin:
                return False
            if arrow.cx < edge or arrow.cx >= width - edge:
                return False
        return True

    def play_level_fast(self, level: int = 0, dry_run: bool = False) -> str:
        """Plan the board once, tap batches, verify between batches."""
        self._hearts_seen = 0
        started = time.monotonic()

        detection = self.read_stable()
        if detection is None:
            return "unstable"

        # step 1: the whole board must be on screen before anything is tapped
        detection = self.ensure_zoom(detection)
        if detection is None:
            return "unstable"
        if not detection.arrows:
            return "empty"

        hearts_ok, red = self.check_hearts(detection.image)
        if not hearts_ok:
            self.log.error("a heart was lost (%d red pixels left)", red)
            self.dump(detection, [], level, "life_lost")
            return "life_lost"

        if dry_run:
            self.report_plan(detection, self.candidates(detection), level)
            return "dryrun"

        if not self.all_tappable(detection):
            self.log.warning("some arrows are uncertain or at the screen edge -> "
                             "using one tap per screenshot for this level")
            return self.play_level_slow(level, dry_run=dry_run)

        first_count = len(detection.arrows)
        #: the first tap of a level is always sent alone, so the camera's actual
        #: behaviour is measured before anything is batched
        batch_limit = 1 if self.config.adaptive_batches else self.config.batch_size
        batches = int((first_count + batch_limit - 1) / batch_limit)
        self.log.info("fast mode: %d arrows, up to %d per batch (start %d, ~%d screenshots)",
                      first_count, self.config.batch_size, batch_limit, batches)

        stalls = 0
        batch_index = 0
        previous_count = first_count
        while detection.arrows:
            if batch_index >= self.config.max_waves_per_level:
                self.log.error("stopped after %d batches on this level", batch_index)
                return "exhausted"

            order = solver.order(detection.arrows)
            if order is None:
                self.stats.deadlocks += 1
                self.log.warning("dead-locked: %d arrows but none can move (%s)",
                                 len(detection.arrows), detection.describe())
                self.dump(detection, [], level, "deadlock")
                if self.config.on_deadlock != "tap_anyway":
                    return "deadlock"
                order = [max(detection.arrows, key=lambda arrow: arrow.confidence)]
                self.log.warning("on_deadlock=tap_anyway -> sacrificing %s", order[0])

            batch = order[:batch_limit]
            self.log.debug("batch %d: %d arrows left, tapping %d: %s",
                           batch_index + 1, len(detection.arrows), len(batch),
                           ", ".join(str(arrow) for arrow in batch))
            before = detection
            self.tap_batch(batch)
            self._sleep(self.config.batch_wait_ms / 1000.0)

            after = self.detect_once()
            if after is None:
                self.log.error("could not read the board after batch %d", batch_index + 1)
                return "error"
            detection = after
            batch_index += 1

            hearts_ok, red = self.check_hearts(detection.image)
            if not hearts_ok:
                self.log.error("a heart was lost during batch %d (%d red pixels left)",
                               batch_index, red)
                self.dump(detection, [], level, "life_lost")
                return "life_lost"

            count = len(detection.arrows)
            if count > previous_count:
                if previous_count <= len(batch):
                    # this batch covered every arrow, so the board was cleared and
                    # the next level loaded before the screenshot
                    self.log.info("level %d cleared (next board already loaded: %d arrows)",
                                  level, count)
                    return "cleared"
                self.log.warning("%d new arrow(s) appeared mid-level", count - previous_count)
                self.dump(detection, [], level, "changed")
                return "changed"
            if count == previous_count:
                stalls += 1
                if stalls > self.config.tap_retries:
                    self.log.warning("taps have no effect (%d arrows)", count)
                    self.dump(detection, [], level, "stalled")
                    return "stalled"
            else:
                stalls = 0
                if count > previous_count - len(batch):
                    # the removal animation was still running when we looked
                    self.log.debug("board still settling (%d -> %d arrows after %d taps)",
                                   previous_count, count, len(batch))
                    self._sleep(self.config.batch_wait_ms / 1000.0)
                    settled = self.detect_once()
                    if settled is not None and len(settled.arrows) <= count:
                        detection = settled
                        hearts_ok, _ = self.check_hearts(detection.image)
                        if not hearts_ok:
                            return "life_lost"
            previous_count = len(detection.arrows)

            if self.config.adaptive_batches:
                drift = self.camera_drift(before, detection)
                if drift is None:
                    pass                      # too few survivors to judge
                elif drift > self.config.batch_drift_px:
                    # a moving camera invalidates coordinates within a batch: only
                    # batch again once it has been seen standing still
                    if drift > 2 * self.config.batch_drift_px:
                        batch_limit = 1
                    else:
                        batch_limit = max(1, batch_limit // 2)
                    self.log.debug("camera moved %.1f px during the batch - batch size down to %d",
                                   drift, batch_limit)
                else:
                    batch_limit = min(self.config.batch_size,
                                      max(self.config.batch_size_start, batch_limit * 2))
                    self.log.debug("camera steady - next batch size %d", batch_limit)

            # the game zooms in as the board empties; if that pushed ink off the
            # board region, pinch back out before the next batch
            if detection.arrows and self.board_clipped(detection):
                self.log.info("camera zoomed in mid-level - zooming out again")
                fitted = self.ensure_zoom(detection)
                if fitted is not None:
                    detection = fitted
                    previous_count = len(detection.arrows)
                if self.board_clipped(detection):
                    self.log.warning("still clipped after a pinch -> one tap per screenshot")
                    return self.play_level_slow(level, dry_run=dry_run)
            if not self.all_tappable(detection):
                self.log.warning("arrows reached the screen edge -> one tap per screenshot")
                return self.play_level_slow(level, dry_run=dry_run)

        elapsed = time.monotonic() - started
        self.log.info("level %d cleared in %.1f s (%d taps, %.1f arrows/s)",
                      level, elapsed, self.stats.taps - self._level_start_taps,
                      first_count / elapsed if elapsed > 0 else 0.0)
        return "cleared"

    def play_level_slow(self, level: int = 0, dry_run: bool = False) -> str:
        """Clear one board.  Returns ``cleared``, ``dryrun`` or a failure reason."""
        self._hearts_seen = 0
        steps = 0
        previous_count: Optional[int] = None
        stalls = 0

        while steps < self.config.max_waves_per_level:
            detection = self.read_stable()
            if detection is None:
                return "unstable"
            if steps == 0:
                detection = self.ensure_zoom(detection)
                if detection is None:
                    return "unstable"
            count = len(detection.arrows)
            if count == 0:
                return "cleared" if steps else "empty"

            hearts_ok, red = self.check_hearts(detection.image)
            if not hearts_ok:
                self.log.error("a heart was lost (%d red pixels left)", red)
                self.dump(detection, [], level, "life_lost")
                return "life_lost"

            if previous_count is not None:
                if count > previous_count:
                    self.log.warning(
                        "%d new arrow(s) appeared mid-level - a popup or ad is in the way",
                        count - previous_count,
                    )
                    self.dump(detection, [], level, "changed")
                    return "changed"
                if count == previous_count:
                    stalls += 1
                    if stalls > self.config.tap_retries:
                        self.log.warning("taps have no effect (%d arrows)", count)
                        self.dump(detection, [], level, "stalled")
                        return "stalled"
                else:
                    stalls = 0

            safe = self.candidates(detection)
            if dry_run:
                self.report_plan(detection, safe, level)
                return "dryrun"
            if not safe:
                self.stats.deadlocks += 1
                self.log.warning(
                    "dead-locked: %d arrows but none can move (%s)",
                    count, detection.describe(),
                )
                self.dump(detection, [], level, "deadlock")
                if self.config.on_deadlock != "tap_anyway":
                    return "deadlock"
                safe = [max(detection.arrows, key=lambda arrow: arrow.confidence)]
                self.log.warning("on_deadlock=tap_anyway -> sacrificing %s", safe[0])

            batch = safe[: self.config.chunk_size]
            self.log.info(
                "step %d: %d arrows, %d safe%s -> tapping %s",
                steps + 1, count, len(safe),
                "" if len(batch) == 1 else f", using {len(batch)}",
                ", ".join(str(arrow) for arrow in batch),
            )
            for arrow in batch:
                self.tap(arrow)
            previous_count = count
            steps += 1
            self._sleep(self.config.tap_wait_ms / 1000.0)

        self.log.error("stopped after %d taps on this level", steps)
        return "exhausted"

    # ------------------------------------------------------------------- flow
    def advance_level(self, level: int) -> bool:
        """Wait for the next board (the game advances by itself)."""
        timeout = self.config.advance_timeout_ms / 1000.0
        wait = self.config.advance_wait_ms / 1000.0
        started = time.monotonic()
        taps = 0
        while time.monotonic() - started < timeout:
            detection = self.read_stable()
            if self.looks_like_board(detection):
                self.log.info("next level loaded (%d arrows)", len(detection.arrows))
                return True
            elapsed = time.monotonic() - started
            if taps < self.config.advance_attempts and elapsed >= (taps + 1) * wait:
                point = self.config.next_tap_points[taps % len(self.config.next_tap_points)]
                image_height, image_width = self._frame_shape or (2392, 1080)
                x, y = int(round(point[0] * image_width)), int(round(point[1] * image_height))
                self.log.debug("fallback advance tap %d at (%d, %d)", taps + 1, x, y)
                self.device.tap(x, y)
                self.stats.taps += 1
                taps += 1
        self.log.error("no new board within %.1f s", timeout)
        return False

    def run(self, dry_run: bool = False) -> Stats:
        cleared = 0
        level = int(self.config.start_level)

        try:
            while cleared < self.config.max_levels:
                level_started = time.monotonic()
                self._level_start_taps = self.stats.taps
                self.log.info("--- level %d (%d/%d cleared) ---",
                              level, cleared, self.config.max_levels)
                try:
                    outcome = self.play_level(level, dry_run=dry_run)
                except KeyboardInterrupt:
                    raise
                except Exception as exc:  # keep the run alive on transient errors
                    self.stats.errors += 1
                    self.log.exception("level %d crashed: %s", level, exc)
                    if self.config.stop_on_error:
                        raise
                    outcome = "error"

                if outcome == "dryrun":
                    return self.stats

                if outcome == "cleared":
                    self.stats.levels_cleared += 1
                    cleared += 1
                    level += 1
                    level_seconds = time.monotonic() - level_started
                    self.stats.level_times.append(level_seconds)
                    self.log.info("level cleared -> %d total, %.1f s, %d taps",
                                  cleared, level_seconds,
                                  self.stats.taps - self._level_start_taps)
                    if not self.advance_level(level):
                        self.log.error("could not get to the next level - stopping")
                        break
                    continue

                self.stats.failed_levels += 1
                self.log.warning("level %d did not clear (%s): %s", level, outcome,
                                 _HINTS.get(outcome, "see the log above"))
                if not self.config.continue_on_failure:
                    break
                if not self.advance_level(level):
                    break
                level += 1
        except KeyboardInterrupt:
            self.log.warning("interrupted by user")

        return self.stats

    # ------------------------------------------------------------------ debug
    def report_plan(self, detection: Detection, safe: Sequence[Arrow], level: int) -> None:
        """Dry-run output: what the bot sees and what it would do."""
        self.log.info("detection: %s", detection.describe())
        for arrow in sorted(detection.arrows, key=lambda a: (a.cy, a.cx)):
            self.log.info("  %s tap=%s", arrow, arrow.tap_point())
        low = [arrow for arrow in detection.arrows
               if arrow.confidence < self.config.min_direction_margin]
        if low:
            self.log.warning("%d arrow(s) too ambiguous to tap: %s", len(low),
                             ", ".join(str(arrow) for arrow in low))
        waves = detection.plan()
        if waves is None:
            self.log.warning("this board dead-locks (or an arrow was misread)")
        else:
            self.log.info("plan: %d wave(s), %d tap(s)", len(waves),
                          sum(len(wave) for wave in waves))
        self.dump(detection, safe, level, "dryrun")

    def dump(self, detection: Detection, safe: Sequence[Arrow],
             level: int, tag: str) -> None:
        if not self.config.write_debug:
            return
        try:
            os.makedirs(self.config.debug_dir, exist_ok=True)
            path = os.path.join(self.config.debug_dir, f"L{level:03d}_{tag}.png")
            overlay = render_overlay(
                detection.image, detection, safe=safe,
                title=f"L{level} {tag} | {detection.describe()}",
            )
            cv2.imwrite(path, overlay)
            self.log.info("saved debug image %s", path)
        except Exception as exc:  # never let logging kill a run
            self.log.debug("could not write debug image: %s", exc)
