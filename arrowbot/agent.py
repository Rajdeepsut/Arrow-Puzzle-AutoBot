"""The level loop: read the board, tap safe arrows, verify, repeat.

Two modes share the same safety rule (*an arrow may only be tapped when nothing
is in its corridor*):

* **fast** (default) - the whole board is planned once, then tapped in batches
  over scrcpy.  Removing an arrow can never block another one, so every arrow in
  that plan is safe in that exact order; a screenshot between batches catches a
  panned camera and lost hearts.  At the default view the board does not move at
  all between taps, which makes batching safe (measured: 0.0 px on 73 heads).
* **slow** - one tap per screenshot, for a zoomed-in or unstable camera.

A real level is always solvable, so a mathematical deadlock almost always means
a glyph was misread.  :meth:`ArrowAgent.play_level` therefore does not stop on
a deadlock/popup/lost heart: it re-reads the frame, repairs the most ambiguous
directions, tries alternate ink thresholds and slow mode, dismisses popups via
``uiautomator`` and finally relaunches the game.  The run loop keeps coming back
to the same level until it is cleared - it never advances on an uncleared board
and never gives up on one.

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

from . import counter, solver
from .config import Config
from .vision import Arrow, Detection, Detector, fractions_to_box, render_overlay


class Device(Protocol):
    """Anything that can show a screen and accept taps (adb or the simulator)."""

    def screencap(self) -> np.ndarray: ...

    def tap(self, x: int, y: int) -> None: ...


#: what to tell the user when a level does not finish cleanly
_HINTS = {
    "empty": "no arrows on screen - the game is on a menu, a win screen or an ad",
    "unstable": "the camera never stopped moving - a popup or an ad is animating",
    "changed": "something new appeared on the board - a popup or ad is covering it",
    "stalled": "taps had no visible effect - a dialog is probably waiting",
    "deadlock": "no arrow can move (misread arrow, or a special tile)",
    "life_lost": "a heart was lost (a blocked arrow was tapped)",
    "exhausted": "gave up after 'max_waves_per_level'",
    "error": "unexpected error - see the traceback above",
    "advance": "the next level did not load after a clear",
}


@dataclass
class Stats:
    levels_cleared: int = 0
    failed_levels: int = 0
    taps: int = 0
    deadlocks: int = 0
    hearts_lost: int = 0
    errors: int = 0
    retries: int = 0
    recoveries: int = 0
    restarts: int = 0
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
            f"hearts lost: {self.hearts_lost} | retries: {self.retries} | "
            f"recoveries: {self.recoveries} | restarts: {self.restarts} | "
            f"errors: {self.errors} | "
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
        self._base_detector = self.detector
        self.log = logger or logging.getLogger("arrowbot")
        self.stats = Stats()
        self._sleep = time.sleep
        self._hearts_seen = 0
        self._frame_shape: Optional[Tuple[int, int]] = None
        #: which pinch direction worked last time ("out" = fingers together)
        self._zoom_direction = "out"
        self._level_start_taps = 0
        #: the next board, already read by the previous level's last screenshot
        self._pending_detection: Optional[Detection] = None
        #: the last board seen before the final tapping batch (advance guard)
        self._cleared_view: Optional[Detection] = None
        #: boards seen during the current level, to tell a popup from a new level
        self._seen_views: List[Detection] = []
        #: recovery bookkeeping
        self._allow_sacrifice = False
        self._sacrifices_this_level = 0
        self._sacrifice_level: Optional[int] = None
        self._failure_streak = 0

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
            try:
                image = self.device.screencap()
            except Exception as exc:      # a dropped frame / adb hiccup is not fatal
                self.log.debug("screencap failed while settling: %s", exc)
                self._sleep(0.15)
                continue
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

    def _tap_fraction(self, point: Sequence[float]) -> None:
        height, width = self._frame_shape or (2392, 1080)
        self.device.tap(int(round(point[0] * width)), int(round(point[1] * height)))
        self.stats.taps += 1

    # -------------------------------------------------------- device helpers
    def _adb(self):
        """The adb object behind the device, when the backend has one."""
        adb = getattr(self.device, "adb", None)
        if adb is not None and hasattr(adb, "shell"):
            return adb
        if hasattr(self.device, "shell") and hasattr(self.device, "screencap"):
            return self.device
        return None

    def _ui_source(self):
        for candidate in (getattr(self.device, "adb", None), self.device):
            if candidate is not None and callable(getattr(candidate, "find_text", None)):
                return candidate
        return None

    def prepare_device(self) -> None:
        """Wake the phone and keep the screen on, so ADB stays authorised."""
        adb = self._adb()
        if adb is None:
            return
        try:
            if self.config.keep_awake:
                adb.keep_awake()
            adb.wake()
        except Exception as exc:  # never fatal - the phone may already be awake
            self.log.debug("could not wake the device: %s", exc)

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

    # ---------------------------------------------------------------- counter
    def read_counter(self, image: np.ndarray) -> Optional[int]:
        """The HUD's remaining-arrow count, or ``None`` if it is unreadable."""
        if not self.config.counter_check:
            return None
        return counter.read_counter(image, self.config.counter_roi,
                                    self.config.counter_min_score)

    def settle_counter(self, detection: Detection,
                       retries: Optional[int] = None) -> Tuple[Detection, bool]:
        """Wait out a removal/count-up animation before trusting the board.

        Returns the newest detection and whether the HUD count and the detected
        arrows agree.  An unreadable HUD counts as agreement: the check is a
        bonus, never a prerequisite, so an unknown font cannot stop the run.
        """
        if detection is None or not self.config.counter_check:
            return detection, True
        retries = self.config.counter_retries if retries is None else retries
        for attempt in range(retries + 1):
            hud = self.read_counter(detection.image)
            if hud is None or hud == len(detection.arrows):
                if hud is not None:
                    self.log.debug("HUD count agrees (%d arrows)", hud)
                return detection, True
            self.log.debug("HUD says %d arrows, %d were read (attempt %d/%d)",
                           hud, len(detection.arrows), attempt + 1, retries + 1)
            if attempt >= retries:
                break
            self._sleep(max(0.0, self.config.counter_wait_ms / 1000.0))
            fresh = self.detect_once(retries=1)
            if fresh is not None:
                detection = fresh
        return detection, False

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

    def looks_like_board(self, detection: Optional[Detection], minimum: int = 4) -> bool:
        """Is this a playable board rather than a menu, ad or win screen?

        A normal level has many arrows.  The first levels only have a few, so
        when the count is low the hearts HUD is the tie-breaker: it is visible
        on a real level and hidden on menus/ads.
        """
        if detection is None or not len(detection.arrows):
            return False
        if len(detection.arrows) >= minimum:
            return True
        if self.config.hearts_min_red <= 0:
            return False
        return self.red_pixels(detection.image) >= self.config.hearts_min_red

    @staticmethod
    def _same_board(first: Optional[Detection], second: Optional[Detection],
                    tolerance: int = 3) -> bool:
        """True when two detections show the same, unmoved board."""
        if first is None or second is None:
            return False
        return first.same_view(second, tolerance)

    def _stable_next_board(self, candidate: Optional[Detection]) -> Optional[Detection]:
        """Accept the next board only once two consecutive reads agree.

        The screenshot that reveals the next level can catch the arrows while
        they are still flying in, so it is never used as-is.  ``None`` means
        "not loaded yet" and the caller falls back to :meth:`advance_level`.
        """
        if candidate is None or not self.looks_like_board(candidate):
            return None
        if self._same_board(candidate, self._cleared_view):
            return None
        stable = self.read_stable()
        if stable is None or not self.looks_like_board(stable):
            return None
        if self._same_board(stable, self._cleared_view):
            return None
        return stable

    def _remember(self, detection: Detection, limit: int = 4) -> None:
        if not detection.arrows:
            return
        self._seen_views.append(detection)
        if len(self._seen_views) > limit:
            del self._seen_views[:-limit]

    def repair_plan(self, detection: Detection) -> Tuple[Detection, Optional[List[Arrow]]]:
        """Re-read once, then repair the most ambiguous directions.

        A deadlock on a real board means the pixel reading is wrong, not that
        the level is impossible.  The cheapest direction flips that turn the
        board into a clearable one are found first; the caller taps that order.
        """
        fresh = self.detect_once()
        if fresh is not None and fresh.arrows:
            if (len(fresh.arrows) == len(detection.arrows)
                    and self._same_board(detection, fresh, tolerance=4)):
                order = solver.order(fresh.arrows)
                if order is not None:
                    self.log.info("re-read fixed the board (no deadlock)")
                    return fresh, order

        fixes = solver.repair(detection.arrows)
        if fixes:
            for arrow, direction in fixes.items():
                self.log.info("repairing misread arrow %s -> %s", arrow, direction)
                arrow.direction = direction
            order = solver.order(detection.arrows)
            if order is not None:
                self.log.info("repaired a deadlock by fixing %d direction(s)", len(fixes))
                return detection, order
        return detection, None

    # ------------------------------------------------------------------ level
    def play_level(self, level: int = 0, dry_run: bool = False,
                   initial: Optional[Detection] = None) -> str:
        """Clear one board, recovering from every kind of hiccup.

        Returns ``cleared``, ``dryrun`` or the last failure reason.  It tries
        normal fast mode, a fresh read, slow mode, alternate ink thresholds and
        (after a few attempts) a deliberate sacrifice; popups are dismissed with
        ``uiautomator`` between attempts.
        """
        if self._sacrifice_level != level:
            self._sacrifices_this_level = 0
            self._sacrifice_level = level
        self._pending_detection = None
        attempts = max(1, int(self.config.max_level_attempts))
        outcome = "error"
        for attempt in range(1, attempts + 1):
            self._allow_sacrifice = (
                self.config.on_deadlock == "tap_anyway"
                or (self.config.on_deadlock == "repair"
                    and attempt > self.config.sacrifice_after_attempts)
            )
            self.detector = self._detector_for_attempt(attempt)
            slow = (not self.config.fast_mode
                    or (self.config.fast_mode and attempt in (3, 6)))
            start = initial if attempt == 1 else None
            self.log.info("level %d: attempt %d/%d (%s mode%s)",
                          level, attempt, attempts,
                          "slow" if slow else "fast",
                          ", sacrifice allowed" if self._allow_sacrifice else "")
            try:
                if slow:
                    outcome = self.play_level_slow(level, dry_run=dry_run, initial=start)
                else:
                    outcome = self.play_level_fast(level, dry_run=dry_run, initial=start)
            except KeyboardInterrupt:
                raise
            except Exception as exc:  # keep the run alive on transient errors
                self.stats.errors += 1
                self.log.exception("level %d crashed: %s", level, exc)
                outcome = "error"

            if outcome in ("cleared", "dryrun"):
                self.detector = self._base_detector
                return outcome
            self.stats.retries += 1
            self.log.warning("level %d attempt %d failed (%s): %s", level, attempt,
                             outcome, _HINTS.get(outcome, "see the log above"))
            if dry_run:
                break
            self._safe_recover(outcome, level, attempt)
            if self._pending_detection is not None:
                initial = self._pending_detection
                self._pending_detection = None
        self.detector = self._base_detector
        return outcome

    def _detector_for_attempt(self, attempt: int) -> Detector:
        """Normal detector, with one or two more permissive recovery passes."""
        if attempt <= 2:
            return self._base_detector
        if attempt == 4:
            return Detector(self.config, ink_gray_max=min(
                250, int(self.config.ink_gray_max) + 18))
        if attempt == 5:
            return Detector(self.config, min_arrow_pixels=max(
                8, int(self.config.min_arrow_pixels * 0.5)))
        return self._base_detector

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

    def play_level_fast(self, level: int = 0, dry_run: bool = False,
                        initial: Optional[Detection] = None) -> str:
        """Plan the board once, tap batches, verify between batches."""
        self._hearts_seen = 0
        self._cleared_view = None
        self._seen_views = []
        started = time.monotonic()

        detection = initial or self.read_stable()
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
            self.log.error("a heart was lost before the first tap (%d red pixels left)", red)
            self.dump(detection, [], level, "life_lost")
            return "life_lost"

        if dry_run:
            self.report_plan(detection, self.candidates(detection), level)
            return "dryrun"

        # The HUD count is an independent read of the same board.  A mismatch
        # is usually a removal/count-up animation still running, so wait it out;
        # if it survives, the reading cannot be trusted enough for batch taps.
        detection, counter_ok = self.settle_counter(detection)
        if detection is None:
            return "unstable"
        if not counter_ok:
            self.log.warning("HUD count still disagrees -> one tap per screenshot")
            return self.play_level_slow(level, dry_run=dry_run, initial=detection)

        if not self.all_tappable(detection):
            self.log.warning("some arrows are uncertain or at the screen edge -> "
                             "using one tap per screenshot for this level")
            return self.play_level_slow(level, dry_run=dry_run, initial=detection)

        first_count = len(detection.arrows)
        #: the first tap of a level is always sent alone, so the camera's actual
        #: behaviour is measured before anything is batched
        batch_limit = 1 if self.config.adaptive_batches else self.config.batch_size
        batch_estimate = int((first_count + batch_limit - 1) / batch_limit)
        self.log.info("fast mode: %d arrows, up to %d per batch (start %d, ~%d screenshots)",
                      first_count, self.config.batch_size, batch_limit, batch_estimate)

        stalls = 0
        batch_index = 0
        previous_count = first_count
        while detection.arrows:
            if batch_index >= self.config.max_waves_per_level:
                self.log.error("stopped after %d batches on this level", batch_index)
                return "exhausted"
            self._remember(detection)

            order = solver.order(detection.arrows)
            if order is None:
                detection, order = self.repair_plan(detection)
            if order is None:
                self.stats.deadlocks += 1
                self.log.warning("dead-locked: %d arrows but none can move (%s)",
                                 len(detection.arrows), detection.describe())
                self.dump(detection, [], level, "deadlock")
                if (self._allow_sacrifice
                        and self._sacrifices_this_level < self.config.max_sacrifices_per_level):
                    order = [min(detection.arrows, key=lambda arrow: arrow.confidence)]
                    self._sacrifices_this_level += 1
                    self.log.warning("sacrificing a heart on %s to unstick recovery",
                                     order[0])
                else:
                    return "deadlock"

            batch = order[:batch_limit]
            self.log.debug("batch %d: %d arrows left, tapping %d: %s",
                           batch_index + 1, len(detection.arrows), len(batch),
                           ", ".join(str(arrow) for arrow in batch))
            before = detection
            self._cleared_view = before
            self.tap_batch(batch)
            self._sleep(self.config.batch_wait_ms / 1000.0)

            after = self.detect_once()
            if after is None:
                self.log.error("could not read the board after batch %d", batch_index + 1)
                return "error"
            detection = after
            batch_index += 1

            # A batch removes several arrows at once.  Wait until the HUD count
            # and the board agree so the next plan is built on a settled frame.
            detection, counter_ok = self.settle_counter(detection)
            if not counter_ok:
                self.log.warning("HUD count disagrees after batch %d - "
                                 "one tap per screenshot from here", batch_index)
                batch_limit = 1

            hearts_ok, red = self.check_hearts(detection.image)
            if not hearts_ok:
                self.log.warning("a heart was lost during batch %d (%d red pixels left) "
                                 "- continuing with the remaining hearts",
                                 batch_index, red)
                self.dump(detection, [], level, "life_lost")
                batch_limit = 1          # one tap per screenshot from now on
                if not detection.arrows and red < self.config.hearts_min_red:
                    return "life_lost"

            count = len(detection.arrows)
            if count > previous_count:
                if any(self._same_board(detection, seen)
                       for seen in self._seen_views[:-1]):
                    self.log.info("the same board reappeared (popup gone) - continuing")
                    previous_count = count
                    continue
                if previous_count <= len(batch):
                    # this batch covered every arrow, so the board was cleared and
                    # the next level loaded before the screenshot
                    self.log.info("level %d cleared (next board already loaded: %d arrows)",
                                  level, count)
                    # the next board may still be animating in; only hand a
                    # verified, settled read to the next level
                    self._pending_detection = self._stable_next_board(detection)
                    if self._pending_detection is None:
                        self.log.info("next board is not settled yet - waiting for it")
                    return "cleared"
                self.log.warning("%d new arrow(s) appeared mid-level", count - previous_count)
                self.dump(detection, [], level, "changed")
                return "changed"
            if count == previous_count:
                # one delayed removal can look like "no effect"; read once more
                self._sleep(self.config.batch_wait_ms / 1000.0)
                retry = self.detect_once()
                if retry is not None:
                    detection = retry
                    count = len(detection.arrows)
                    hearts_ok, _ = self.check_hearts(detection.image)
                if count == previous_count:
                    stalls += 1
                    if stalls > self.config.tap_retries:
                        self.log.warning("taps have no effect (%d arrows)", count)
                        self.dump(detection, [], level, "stalled")
                        return "stalled"
                else:
                    stalls = 0
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
                        count = len(detection.arrows)
                        hearts_ok, _ = self.check_hearts(detection.image)
            previous_count = count

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
                    return self.play_level_slow(level, dry_run=dry_run, initial=detection)
            if not self.all_tappable(detection):
                self.log.warning("arrows reached the screen edge -> one tap per screenshot")
                return self.play_level_slow(level, dry_run=dry_run, initial=detection)

        elapsed = time.monotonic() - started
        self.log.info("level %d cleared in %.1f s (%d taps, %.1f arrows/s)",
                      level, elapsed, self.stats.taps - self._level_start_taps,
                      first_count / elapsed if elapsed > 0 else 0.0)
        return "cleared"

    def play_level_slow(self, level: int = 0, dry_run: bool = False,
                        initial: Optional[Detection] = None) -> str:
        """Clear one board with one tap per screenshot (safest mode)."""
        self._hearts_seen = 0
        self._cleared_view = None
        self._seen_views = []
        steps = 0
        previous_count: Optional[int] = None
        stalls = 0
        detection = initial

        while steps < self.config.max_waves_per_level:
            if detection is None:
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
            self._remember(detection)

            hearts_ok, red = self.check_hearts(detection.image)
            if not hearts_ok:
                self.log.warning("a heart was lost (%d red pixels left) - continuing",
                                 red)
                self.dump(detection, [], level, "life_lost")
                if not detection.arrows and red < self.config.hearts_min_red:
                    return "life_lost"

            if previous_count is not None:
                if count > previous_count:
                    if any(self._same_board(detection, seen)
                           for seen in self._seen_views[:-1]):
                        self.log.info("the same board reappeared (popup gone) - continuing")
                    else:
                        self.log.warning(
                            "%d new arrow(s) appeared mid-level - a popup or ad is in the way",
                            count - previous_count,
                        )
                        self.dump(detection, [], level, "changed")
                        return "changed"
                elif count == previous_count:
                    stalls += 1
                    if stalls > self.config.tap_retries:
                        self.log.warning("taps have no effect (%d arrows)", count)
                        self.dump(detection, [], level, "stalled")
                        return "stalled"
                else:
                    stalls = 0

            safe = self.candidates(detection)
            if not safe:
                detection, order = self.repair_plan(detection)
                safe = order or self.candidates(detection)
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
                if (self._allow_sacrifice
                        and self._sacrifices_this_level < self.config.max_sacrifices_per_level):
                    safe = [min(detection.arrows, key=lambda arrow: arrow.confidence)]
                    self._sacrifices_this_level += 1
                    self.log.warning("sacrificing a heart on %s to unstick recovery", safe[0])
                else:
                    return "deadlock"

            batch = safe[: self.config.chunk_size]
            self.log.info(
                "step %d: %d arrows, %d safe%s -> tapping %s",
                steps + 1, count, len(safe),
                "" if len(batch) == 1 else f", using {len(batch)}",
                ", ".join(str(arrow) for arrow in batch),
            )
            self._cleared_view = detection
            for arrow in batch:
                self.tap(arrow)
            previous_count = count
            steps += 1
            detection = None
            self._sleep(self.config.tap_wait_ms / 1000.0)

        self.log.error("stopped after %d taps on this level", steps)
        return "exhausted"

    # ------------------------------------------------------------------- flow
    def advance_level(self, level: int = 0) -> Optional[Detection]:
        """Wait for the next board (the game advances by itself).

        Returns the fresh, stable detection or ``None``.  A board identical to
        the one just cleared is rejected, so a popup that temporarily hid the
        arrows can never be mistaken for a new level.
        """
        timeout = self.config.advance_timeout_ms / 1000.0
        wait = self.config.advance_wait_ms / 1000.0
        started = time.monotonic()
        taps = 0
        # a fast auto-advance needs no accessibility dump; only start looking
        # for overlay buttons after the board has had a moment to load
        last_ui = time.monotonic()
        while time.monotonic() - started < timeout:
            stable = self.read_stable()
            visible = self.looks_like_board(stable)
            if visible and not self._same_board(stable, self._cleared_view):
                self.log.info("next level loaded (%d arrows)", len(stable.arrows))
                return stable

            now = time.monotonic()
            if now - last_ui >= 1.2:
                last_ui = now
                if self.dismiss_overlays():
                    continue

            elapsed = now - started
            if not visible and taps < self.config.advance_attempts \
                    and elapsed >= (taps + 1) * wait:
                point = self.config.next_tap_points[
                    taps % len(self.config.next_tap_points)]
                self.log.debug("fallback advance tap %d at %s", taps + 1, point)
                self._tap_fraction(point)
                taps += 1
                self._sleep(0.1)
            else:
                self._sleep(0.15)

        self.log.error("no new board within %.1f s", timeout)
        return None

    # --------------------------------------------------------------- recovery
    def dismiss_overlays(self) -> bool:
        """Press a ``Continue``/``Retry``/``Next`` button if one is on screen."""
        if not self.config.ui_recovery:
            return False
        source = self._ui_source()
        if source is None:
            return False
        try:
            nodes = source.find_text(self.config.ui_labels)
        except Exception as exc:
            self.log.debug("uiautomator failed: %s", exc)
            return False
        if not nodes:
            return False
        node = nodes[0]
        x, y = node.center
        self.log.info("pressing overlay button %r at (%d, %d)", node.label, x, y)
        self.device.tap(x, y)
        self.stats.taps += 1
        self._sleep(0.4)
        return True

    def blind_recovery(self) -> bool:
        """Tap through an ad/home screen when no board (and no UI label) is visible.

        Only used when the detector cannot see a board, so these taps cannot hit
        a real arrow.
        """
        groups = (self.config.overlay_tap_points, self.config.home_tap_points)
        for points in groups:
            for point in points:
                self.log.info("blind recovery tap at %s", point)
                self._tap_fraction(point)
                self._sleep(0.45)
                detection = self.detect_once(retries=1)
                if self.looks_like_board(detection):
                    return True
        return False

    def restart_app(self) -> bool:
        """Relaunch the game; the level is saved and resumes where it was."""
        adb = self._adb()
        package = self.config.game_package
        if adb is None or not package:
            return False
        self.log.warning("restarting %s to clear a stuck screen", package)
        self.stats.restarts += 1
        try:
            adb.force_stop(package)
            self._sleep(0.5)
            try:
                adb.start_app(package, self.config.game_activity or None)
            except Exception:
                adb.start_app(package, None)
            self._sleep(max(0.5, self.config.restart_wait_ms / 1000.0))
            adb.wake()
            self.dismiss_overlays()
            detection = self.detect_once(retries=2)
            if not self.looks_like_board(detection):
                self.blind_recovery()
            return True
        except Exception as exc:
            self.log.warning("could not restart the game: %s", exc)
            return False

    def recover(self, outcome: str, level: int, attempt: int = 1) -> None:
        """Do whatever is needed before the next attempt at the same level."""
        self.stats.recoveries += 1
        self._failure_streak += 1
        base = self.config.retry_wait_ms / 1000.0
        self._sleep(min(2.0, base * min(4, max(1, attempt))))

        if self.dismiss_overlays():
            return

        detection = self.detect_once(retries=1)
        if not self.looks_like_board(detection):
            if self.blind_recovery():
                return

        if self._failure_streak % max(1, self.config.restart_after_failures) == 0:
            self.restart_app()

    def _safe_recover(self, outcome: str, level: int, attempt: int = 1) -> None:
        """A failed recovery must not kill a long run either."""
        try:
            self.recover(outcome, level, attempt)
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            self.stats.errors += 1
            self.log.exception("recovery for level %d failed: %s", level, exc)

    def _safe_advance(self, level: int) -> Optional[Detection]:
        """Waiting for the next board must survive a dropped tap/socket."""
        try:
            return self.advance_level(level)
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            self.stats.errors += 1
            self.log.exception("waiting for the next level failed: %s", exc)
            return None

    # -------------------------------------------------------------------- run
    def run(self, dry_run: bool = False) -> Stats:
        self.prepare_device()
        cleared = 0
        level = int(self.config.start_level)
        pending: Optional[Detection] = None

        try:
            while cleared < self.config.max_levels:
                level_started = time.monotonic()
                self._level_start_taps = self.stats.taps
                self.log.info("--- level %d (%d/%d cleared) ---",
                              level, cleared, self.config.max_levels)
                outcome = self.play_level(level, dry_run=dry_run, initial=pending)
                pending = self._pending_detection
                self._pending_detection = None

                if outcome == "dryrun":
                    return self.stats

                if outcome == "cleared":
                    self.stats.levels_cleared += 1
                    cleared += 1
                    self._failure_streak = 0
                    level_seconds = time.monotonic() - level_started
                    self.stats.level_times.append(level_seconds)
                    self.log.info("level cleared -> %d total, %.1f s, %d taps",
                                  cleared, level_seconds,
                                  self.stats.taps - self._level_start_taps)

                    # Only a genuinely cleared board gets here.  If the next board
                    # was not already in the verification screenshot, wait for it.
                    if pending is None:
                        pending = self._safe_advance(level)
                        if pending is None:
                            self.log.warning("next board did not appear - recovering")
                            for _ in range(2):
                                self._safe_recover("advance", level, 1)
                                pending = self._safe_advance(level)
                                if pending is not None:
                                    break
                    level += 1
                    continue

                self.stats.failed_levels += 1
                self.log.warning("level %d did not clear (%s): %s", level, outcome,
                                 _HINTS.get(outcome, "see the log above"))
                if not self.config.continue_on_failure:
                    break
                # never advance on an uncleared board: recover and play it again
                self._safe_recover(outcome, level, self.config.max_level_attempts)
                pending = None
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
