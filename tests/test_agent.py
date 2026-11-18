"""End-to-end agent tests: different board sizes, zoomed starts, slow fallback."""

from __future__ import annotations

import logging
import time
from dataclasses import replace

import cv2
import numpy as np
import pytest

from arrowbot.adb import UiNode
from arrowbot.agent import ArrowAgent
from arrowbot.config import Config
from arrowbot.selftest import make_arrow
from arrowbot.sim import PITCH, SimGame
from arrowbot.vision import Detector, fractions_to_box


def _config(levels: int) -> Config:
    config = Config()
    config.write_debug = False
    config.max_levels = levels
    config.fast_mode = True
    config.batch_size = 8
    config.batch_size_start = 6
    config.batch_gap_ms = 0
    config.batch_wait_ms = 0
    config.tap_hold_ms = 0
    config.zoom_settle_ms = 0
    config.settle_poll_ms = 10
    config.settle_timeout_ms = 2000
    config.advance_wait_ms = 20
    config.advance_timeout_ms = 3000
    config.advance_attempts = 0
    config.retry_wait_ms = 0
    config.restart_wait_ms = 0
    config.restart_after_failures = 999
    return config


def _logger() -> logging.Logger:
    logger = logging.getLogger("arrowbot.tests")
    logger.handlers = [logging.NullHandler()]
    logger.propagate = False
    logger.setLevel(logging.CRITICAL)
    return logger


class _PopupSim(SimGame):
    """A sim that starts behind a fake "CONTINUE" popup once."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.popup = True
        self.dismissed = 0
        self.button = (int(self.width * 0.28), int(self.height * 0.45),
                       int(self.width * 0.72), int(self.height * 0.56))

    def screencap(self):
        image = super().screencap()
        if self.popup:
            x0, y0, x1, y1 = self.button
            cv2.rectangle(image, (x0 - 40, y0 - 90), (x1 + 40, y1 + 40),
                          (245, 245, 245), -1)
            cv2.putText(image, "CONTINUE", (x0, y1 - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (40, 40, 40), 2, cv2.LINE_AA)
        return image

    def find_text(self, patterns):
        if self.popup and any(pattern in "continue" for pattern in patterns):
            return [UiNode("CONTINUE", "", self.button, clickable=True)]
        return []

    def tap(self, x, y):
        if self.popup:
            x0, y0, x1, y1 = self.button
            if x0 <= x <= x1 and y0 <= y <= y1:
                self.popup = False
                self.dismissed += 1
            return
        super().tap(x, y)


class _MenuSim(SimGame):
    """Starts on a blank menu; one blind "Continue Game" tap opens the board."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.menu = True
        self.menu_taps = 0

    def screencap(self):
        if not self.menu:
            return super().screencap()
        image = np.empty((self.height, self.width, 3), np.uint8)
        image[:] = self.background
        self._draw_hud(image)
        # the "Continue Game" button is blank paper: no arrow-like ink in the ROI
        return image

    def tap(self, x, y):
        if self.menu:
            if (abs(x - 0.5 * self.width) < 0.2 * self.width
                    and abs(y - 0.744 * self.height) < 0.06 * self.height):
                self.menu = False
                self.menu_taps += 1
            return
        super().tap(x, y)


class _HeartLossSim(SimGame):
    """Loses one visible heart after the first successful tap, like a misread."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.injected = False

    def tap(self, x, y):
        before = len(self.board)
        super().tap(x, y)
        if not self.injected and len(self.board) < before and self.lives > 1:
            self.injected = True
            self.lives -= 1
            self.lives_lost += 1


class _MisreadDetector(Detector):
    """Reads the topmost upward arrow as downward, like a bad template vote."""

    def detect(self, image):
        detection = super().detect(image)
        upward = [arrow for arrow in detection.arrows if arrow.direction == "U"]
        if upward:
            arrow = min(upward, key=lambda item: item.cy)
            arrow.direction = "D"
            arrow.confidence = 0.20
            arrow.scores = {"U": 0.62, "D": 0.55, "R": 0.10, "L": 0.05}
        return detection


@pytest.mark.parametrize("cols,rows,seed", [(9, 11, 1), (11, 13, 2), (14, 16, 3), (7, 9, 4)])
def test_fast_agent_clears_different_levels(cols, rows, seed):
    config = _config(2)
    game = SimGame(config, cols=cols, rows=rows, seed=seed)
    agent = ArrowAgent(config, game, logger=_logger())

    stats = agent.run()
    assert stats.levels_cleared == 2, stats.summary()
    assert game.lives_lost == 0
    assert len(stats.level_times) == 2
    assert max(stats.level_times) < 60.0


def test_fast_agent_recovers_from_a_zoomed_camera():
    config = _config(2)
    game = SimGame(config, cols=11, rows=13, seed=8)
    game.camera_zoom = 2.4
    agent = ArrowAgent(config, game, logger=_logger())

    stats = agent.run()
    assert stats.levels_cleared == 2, stats.summary()
    assert game.lives_lost == 0
    assert game.camera_zoom == 1.0


def test_slow_mode_still_clears_a_level():
    config = _config(1)
    config.fast_mode = False
    config.tap_wait_ms = 0
    config.chunk_size = 1
    game = SimGame(config, cols=9, rows=10, seed=12)
    agent = ArrowAgent(config, game, logger=_logger())

    stats = agent.run()
    assert stats.levels_cleared == 1, stats.summary()
    assert game.lives_lost == 0


def test_batch_taps_are_safe_on_a_tight_board():
    """A big board with long chains must never lose a heart in fast mode."""
    config = _config(1)
    game = SimGame(config, cols=16, rows=18, seed=4)
    arrows = len(game.board)
    agent = ArrowAgent(config, game, logger=_logger())

    stats = agent.run()
    assert stats.levels_cleared == 1, stats.summary()
    assert game.lives_lost == 0
    assert stats.taps >= arrows


def test_camera_drift_sees_a_moved_board():
    config = _config(1)
    game = SimGame(config, cols=11, rows=13, seed=3)
    agent = ArrowAgent(config, game, logger=_logger())
    detector = Detector(config)
    before = detector.detect(game.screencap())

    assert agent.camera_drift(before, before) == pytest.approx(0.0)

    shifted = replace(before, arrows=[replace(arrow, head=(arrow.cx + 7, arrow.cy + 5))
                                      for arrow in before.arrows])
    drift = agent.camera_drift(before, shifted)
    assert drift is not None and drift > config.batch_drift_px


def test_agent_dismisses_a_popup_and_keeps_playing():
    config = _config(2)
    game = _PopupSim(config, cols=9, rows=11, seed=6)
    agent = ArrowAgent(config, game, logger=_logger())

    stats = agent.run()
    assert stats.levels_cleared == 2, stats.summary()
    assert game.dismissed == 1
    assert stats.recoveries >= 1
    assert game.lives_lost == 0


def test_a_lost_heart_does_not_stop_the_run():
    config = _config(2)
    game = _HeartLossSim(config, cols=9, rows=11, seed=9)
    agent = ArrowAgent(config, game, logger=_logger())

    stats = agent.run()
    assert stats.levels_cleared == 2, stats.summary()
    assert game.lives_lost == 1
    assert stats.hearts_lost == 1
    assert stats.failed_levels == 0


def test_agent_repairs_a_misread_and_clears_without_a_heart_loss():
    config = _config(1)
    game = SimGame(config, cols=9, rows=11, seed=13)
    game.board = [make_arrow((300, 300 + 3 * PITCH * index), "U") for index in range(3)]
    game._level_size = len(game.board)
    detector = _MisreadDetector(config)
    agent = ArrowAgent(config, game, detector=detector, logger=_logger())

    stats = agent.run()
    assert stats.levels_cleared == 1, stats.summary()
    assert game.lives_lost == 0
    assert stats.deadlocks == 0


def test_a_failed_level_is_retried_not_skipped():
    """A blank menu fails one attempt; the run recovers and plays the same level."""
    config = _config(1)
    config.max_level_attempts = 1
    game = _MenuSim(config, cols=9, rows=11, seed=21)
    agent = ArrowAgent(config, game, logger=_logger())

    stats = agent.run()
    assert game.menu is False and game.menu_taps == 1
    assert stats.levels_cleared == 1, stats.summary()
    assert stats.failed_levels == 1
    assert game.lives_lost == 0


def test_a_small_early_level_is_recognised_by_its_hearts_hud():
    config = _config(1)
    game = SimGame(config, cols=9, rows=11, seed=4)
    game.board = game.board[:2]
    agent = ArrowAgent(config, game, logger=_logger())
    detection = agent.detect_once()
    assert detection is not None and len(detection.arrows) == 2
    assert agent.looks_like_board(detection)

    stripped = detection.image.copy()
    height, width = stripped.shape[:2]
    x0, y0, x1, y1 = fractions_to_box(config.hearts_roi, width, height)
    stripped[y0:y1, x0:x1] = 255
    assert not agent.looks_like_board(replace(detection, image=stripped))


def test_the_next_board_must_be_settled_before_use():
    config = _config(1)
    game = SimGame(config, cols=9, rows=11, seed=3)
    agent = ArrowAgent(config, game, logger=_logger())
    detection = agent.detect_once()
    assert detection is not None and detection.arrows
    agent._cleared_view = None
    assert agent._stable_next_board(detection) is not None

    agent.read_stable = lambda: None
    assert agent._stable_next_board(detection) is None

    agent.read_stable = lambda: detection
    agent._cleared_view = detection
    assert agent._stable_next_board(detection) is None


def test_a_popup_that_hides_the_board_is_not_a_new_level():
    config = _config(1)
    config.advance_timeout_ms = 400
    config.advance_wait_ms = 10
    config.advance_attempts = 0
    config.ui_recovery = False
    game = SimGame(config, cols=9, rows=11, seed=2)
    game.advance_delay = 60.0
    agent = ArrowAgent(config, game, logger=_logger())
    agent._cleared_view = agent.detect_once()

    started = time.monotonic()
    assert agent.advance_level() is None
    assert time.monotonic() - started < 2.0
