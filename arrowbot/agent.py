"""The level loop: capture -> detect -> solve -> tap -> repeat."""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import List, Optional, Protocol, Sequence, Set, Tuple

import cv2
import numpy as np

from . import solver
from .config import Config
from .vision import Arrow, Detection, Detector, render_overlay

Cell = Tuple[int, int]


class Device(Protocol):
    """Anything that can show a screen and accept taps (adb or the simulator)."""

    def screencap(self) -> np.ndarray: ...

    def tap(self, x: int, y: int) -> None: ...


#: what to tell the user when a level does not finish cleanly
_HINTS = {
    "empty": "no arrows on screen - open the game on a level, or widen 'roi'",
    "changed": "a popup or ad covered the board - close it and run again",
    "stalled": "taps had no visible effect - a dialog is probably waiting",
    "deadlock": "a glyph was misread or the level is stuck - check the debug image",
    "exhausted": "gave up after 'max_waves_per_level'",
    "error": "unexpected error - see the traceback above",
}


def frames_similar(first: np.ndarray, second: np.ndarray, threshold: float) -> bool:
    """True when two screenshots are practically identical (animation done)."""
    if first is None or second is None:
        return False
    if first.shape != second.shape:
        return False
    left = first if first.ndim == 2 else cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
    right = second if second.ndim == 2 else cv2.cvtColor(second, cv2.COLOR_BGR2GRAY)
    left = cv2.resize(left, (160, 90), interpolation=cv2.INTER_AREA)
    right = cv2.resize(right, (160, 90), interpolation=cv2.INTER_AREA)
    return float(np.mean(cv2.absdiff(left, right))) <= threshold


@dataclass
class Stats:
    levels_cleared: int = 0
    failed_levels: int = 0
    waves: int = 0
    taps: int = 0
    deadlocks: int = 0
    errors: int = 0
    started: float = field(default_factory=time.time)

    @property
    def elapsed(self) -> float:
        return time.time() - self.started

    def summary(self) -> str:
        minutes = self.elapsed / 60.0
        rate = self.levels_cleared / minutes if minutes > 0 else 0.0
        return (
            f"levels cleared: {self.levels_cleared} | failed: {self.failed_levels} | "
            f"waves: {self.waves} | taps: {self.taps} | deadlocks: {self.deadlocks} | "
            f"errors: {self.errors} | time: {minutes:.1f} min ({rate:.1f} levels/min)"
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

    # ------------------------------------------------------------------ input
    def capture_settled(self) -> np.ndarray:
        """Capture frames until the screen stops changing."""
        previous = self.device.screencap()
        deadline = time.monotonic() + self.config.settle_timeout_ms / 1000.0
        while time.monotonic() < deadline:
            self._sleep(self.config.settle_poll_ms / 1000.0)
            current = self.device.screencap()
            if frames_similar(previous, current, self.config.settle_thresh):
                return current
            previous = current
        return previous

    def read_board(self) -> Tuple[np.ndarray, Detection]:
        image = self.capture_settled()
        return image, self.detector.detect(image)

    def tap(self, arrow: Arrow) -> None:
        x, y = arrow.tap_point()
        self.device.tap(x, y)
        self.stats.taps += 1
        if self.config.tap_delay_ms > 0:
            self._sleep(self.config.tap_delay_ms / 1000.0)

    # ------------------------------------------------------------------ logic
    def safe_arrows(self, detection: Detection) -> List[Arrow]:
        """Arrows that can be tapped right now without losing a life."""
        board = detection.cells  # every arrow blocks, even unreadable ones
        safe: List[Arrow] = []
        for arrow in detection.tappable(self.config.min_orientation_confidence):
            if solver.is_safe(board, arrow.cell):
                safe.append(arrow)
        return safe

    @staticmethod
    def _unexpected(
        detection: Detection, previous: Sequence[Tuple[float, float]]
    ) -> List[Arrow]:
        """Glyphs that sit where no arrow was in the previous frame.

        Comparing pixels instead of row/column numbers keeps this check immune
        to the lattice being re-fitted between frames.
        """
        radius = max(8.0, 0.5 * detection.spacing)
        limit = radius * radius
        found: List[Arrow] = []
        for arrow in detection.arrows:
            if all(
                (arrow.cx - px) ** 2 + (arrow.cy - py) ** 2 > limit
                for px, py in previous
            ):
                found.append(arrow)
        return found

    # ------------------------------------------------------------------ level
    def play_level(self, level: int = 0, dry_run: bool = False) -> str:
        """Clear one board.

        Returns ``cleared`` (board looks finished), ``deadlock`` (arrows remain
        but nothing is safe), ``stalled`` (taps had no effect), ``changed``
        (something new appeared mid-level, e.g. a popup), ``empty`` (no board
        on screen at all) or ``dryrun``.
        """
        waves = 0
        previous_count: Optional[int] = None
        previous_points: List[Tuple[float, float]] = []
        level_size = 0
        stalls = 0

        while waves < self.config.max_waves_per_level:
            image, detection = self.read_board()
            count = len(detection.arrows)
            level_size = max(level_size, count)

            if count == 0:
                return "cleared" if waves else "empty"

            # Arrows can only ever disappear.  If a glyph shows up where there
            # was no arrow a moment ago, something else is on the screen now:
            # the level-complete dialog, an ad, a popup, a menu.
            if previous_points:
                unexpected = self._unexpected(detection, previous_points)
                if unexpected:
                    remaining = len(previous_points)
                    self.dump(image, detection, set(), level, "changed")
                    if remaining <= max(4, 0.3 * level_size):
                        self.log.info(
                            "%d new glyph(s) on screen but only %d arrows were left "
                            "- assuming the level-complete screen",
                            len(unexpected), remaining,
                        )
                        return "cleared"
                    self.log.warning(
                        "%d new glyph(s) appeared mid-level with %d arrows left "
                        "- a popup or ad is probably covering the board",
                        len(unexpected), remaining,
                    )
                    return "changed"

            if previous_count is not None and count >= previous_count:
                stalls += 1
                self.log.warning(
                    "no visible progress (%d -> %d arrows); retry %d/%d",
                    previous_count, count, stalls, self.config.tap_retries,
                )
                if stalls > self.config.tap_retries:
                    self.dump(image, detection, set(), level, "stalled")
                    if count <= max(4, 0.3 * level_size):
                        self.log.info(
                            "only %d arrow(s) left and taps do nothing - "
                            "assuming the level-complete screen", count,
                        )
                        return "cleared"
                    return "stalled"
            else:
                stalls = 0

            safe = self.safe_arrows(detection)
            if dry_run:
                self.report_plan(image, detection, safe, level)
                return "dryrun"

            if not safe:
                self.stats.deadlocks += 1
                self.log.warning(
                    "dead-locked: %d arrows left but none can be tapped safely", count
                )
                self.log.warning("board:\n%s", detection.to_ascii())
                self.dump(image, detection, set(), level, "deadlock")
                if self.config.on_deadlock != "tap_anyway":
                    return "deadlock"
                victim = max(detection.arrows, key=lambda a: a.confidence)
                self.log.warning("on_deadlock=tap_anyway -> sacrificing %s", victim)
                self.tap(victim)
                previous_count = count
                waves += 1
                continue

            batch = safe[: self.config.chunk_size]
            self.log.info(
                "wave %d: %d arrows, %d safe -> tapping %d",
                waves + 1, count, len(safe), len(batch),
            )
            previous_count = count
            previous_points = [(a.cx, a.cy) for a in detection.arrows]
            for arrow in batch:
                self.log.debug("tap %s at %s", arrow, arrow.tap_point())
                self.tap(arrow)
            waves += 1
            self.stats.waves += 1

        self.log.error("stopped after %d waves on this level", waves)
        return "exhausted"

    def report_plan(
        self, image: np.ndarray, detection: Detection, safe: Sequence[Arrow], level: int
    ) -> None:
        """Dry-run output: show the board, the plan and save an overlay."""
        self.log.info("detection: %s", detection.describe())
        self.log.info("board:\n%s", detection.to_ascii())
        low = [a for a in detection.arrows if not a.known or a.confidence < self.config.min_orientation_confidence]
        if low:
            self.log.warning(
                "%d arrow(s) below the confidence threshold will never be tapped: %s",
                len(low), ", ".join(str(a) for a in low),
            )
        waves = solver.plan(detection.cells)
        if waves is None:
            self.log.warning("this board is dead-locked (or a glyph was misread)")
        else:
            self.log.info(
                "plan: %d wave(s), %d tap(s)",
                len(waves), sum(len(wave) for wave in waves),
            )
            for index, wave in enumerate(waves[:10], 1):
                cells = " ".join(f"r{r}c{c}" for r, c in wave)
                self.log.info("  wave %d: %s", index, cells)
            if len(waves) > 10:
                self.log.info("  ... and %d more wave(s)", len(waves) - 10)
        self.dump(image, detection, {a.cell for a in safe}, level, "dryrun")

    # ------------------------------------------------------------------ flow
    def looks_like_board(self, detection: Detection) -> bool:
        """A real board has several readable arrows, not one stray glyph."""
        return len(detection.tappable(self.config.min_orientation_confidence)) >= 4

    def advance_level(self, level: int) -> bool:
        """Tap through the level-complete screen and confirm the next board."""
        for attempt in range(max(1, self.config.advance_attempts)):
            image = self.capture_settled()
            detection = self.detector.detect(image)
            if self.looks_like_board(detection):
                self.log.info(
                    "next level loaded (%d arrows detected)", len(detection.arrows)
                )
                return True

            point = self.config.next_tap_points[
                attempt % len(self.config.next_tap_points)
            ]
            height, width = image.shape[:2]
            x, y = int(round(point[0] * width)), int(round(point[1] * height))
            self.log.debug("advance tap %d at (%d, %d)", attempt + 1, x, y)
            self.device.tap(x, y)
            self.stats.taps += 1
            self._sleep(self.config.advance_wait_ms / 1000.0)
        return False

    def run(self, dry_run: bool = False) -> Stats:
        cleared = 0
        level = int(self.config.start_level)

        try:
            while cleared < self.config.max_levels:
                self.log.info(
                    "--- level %d (%d/%d cleared) ---", level, cleared, self.config.max_levels
                )
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
                    self.log.info(
                        "level cleared -> %d total, %.1f s, %d taps",
                        cleared, self.stats.elapsed, self.stats.taps,
                    )
                    if not self.advance_level(level):
                        self.log.error(
                            "could not get past the level-complete screen "
                            "(ad, popup or menu?) - stopping"
                        )
                        break
                    continue

                self.stats.failed_levels += 1
                self.log.warning(
                    "level %d did not clear (%s): %s",
                    level, outcome, _HINTS.get(outcome, "see the log above"),
                )
                if not self.config.continue_on_failure:
                    break
                if not self.advance_level(level):
                    break
                level += 1
        except KeyboardInterrupt:
            self.log.warning("interrupted by user")

        return self.stats

    # ------------------------------------------------------------------ debug
    def dump(
        self,
        image: np.ndarray,
        detection: Detection,
        safe: Set[Cell],
        level: int,
        tag: str,
    ) -> None:
        if not self.config.write_debug:
            return
        try:
            os.makedirs(self.config.debug_dir, exist_ok=True)
            path = os.path.join(self.config.debug_dir, f"L{level:03d}_{tag}.png")
            overlay = render_overlay(
                image,
                detection,
                safe_cells=safe,
                min_confidence=self.config.min_orientation_confidence,
                title=f"L{level} {tag} | {detection.describe()}",
            )
            cv2.imwrite(path, overlay)
            self.log.info("saved debug image %s", path)
        except Exception as exc:  # never let logging kill a run
            self.log.debug("could not write debug image: %s", exc)
