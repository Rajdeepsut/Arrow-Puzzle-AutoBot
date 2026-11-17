"""End-to-end agent tests: different board sizes, zoomed starts, slow fallback."""

from __future__ import annotations

import logging
from dataclasses import replace

import pytest

from arrowbot.agent import ArrowAgent
from arrowbot.config import Config
from arrowbot.sim import SimGame
from arrowbot.vision import Detector


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
    return config


def _logger() -> logging.Logger:
    logger = logging.getLogger("arrowbot.tests")
    logger.handlers = [logging.NullHandler()]
    logger.propagate = False
    logger.setLevel(logging.CRITICAL)
    return logger


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
