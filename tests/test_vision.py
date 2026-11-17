"""Vision and zoom tests against the synthetic phone (no real device)."""

from __future__ import annotations

import pathlib

import cv2
import pytest

from arrowbot import solver
from arrowbot.agent import ArrowAgent
from arrowbot.config import Config
from arrowbot.sim import SimGame
from arrowbot.vision import Detector

SAMPLES = pathlib.Path(__file__).resolve().parents[1] / "samples"


def _truth(game: SimGame):
    return [(int(round(game.to_screen(*arrow.path[-1])[0])),
             int(round(game.to_screen(*arrow.path[-1])[1])),
             arrow.direction) for arrow in game.board]


@pytest.mark.parametrize("cols,rows,seed", [(11, 13, 1), (9, 10, 2), (14, 16, 3), (6, 6, 4)])
def test_detection_matches_ground_truth(cols, rows, seed):
    config = Config()
    game = SimGame(config, cols=cols, rows=rows, seed=seed)
    detection = Detector(config).detect(game.screencap())
    assert len(detection.arrows) == len(_truth(game))

    for x, y, direction in _truth(game):
        near = [arrow for arrow in detection.arrows
                if abs(arrow.cx - x) <= 7 and abs(arrow.cy - y) <= 7]
        assert near and near[0].direction == direction
    assert min(arrow.confidence for arrow in detection.arrows) >= 0.06


def test_red_ink_is_detected():
    config = Config()
    game = SimGame(config, cols=11, rows=13, seed=5, foreground=(40, 40, 210))
    detection = Detector(config).detect(game.screencap())
    assert len(detection.arrows) == len(game.board)


def test_a_zoomed_in_camera_clips_the_board():
    config = Config()
    game = SimGame(config, cols=11, rows=13, seed=31)
    full = len(game.board)
    game.camera_zoom = 2.8
    detection = Detector(config).detect(game.screencap())
    assert len(detection.arrows) < full


def test_zoom_out_restores_the_whole_board():
    config = Config()
    config.write_debug = False
    config.zoom_settle_ms = 0
    game = SimGame(config, cols=11, rows=13, seed=31)
    full = len(game.board)
    game.camera_zoom = 2.8
    detector = Detector(config)
    agent = ArrowAgent(config, game)
    clipped = detector.detect(game.screencap())

    fitted = agent.ensure_zoom(clipped)
    assert fitted is not None
    assert not agent.board_clipped(fitted)
    assert len(fitted.arrows) == full
    assert abs(game.camera_zoom - 1.0) < 0.01


def test_a_fitted_view_is_not_pinched():
    config = Config()
    config.write_debug = False
    game = SimGame(config, cols=11, rows=13, seed=2)
    agent = ArrowAgent(config, game)
    detection = agent.ensure_zoom(agent.read_stable())
    assert detection is not None
    assert game.camera_zoom == 1.0


@pytest.mark.parametrize("name,min_arrows,scale_at_least", [
    ("real_level.png", 50, 1.2),          # fitted camera, the normal play state
    ("real_level_zoomed.png", 70, 0.5),   # camera mid-adjustment, scale ~1.0
])
def test_real_game_frames_are_readable(name, min_arrows, scale_at_least):
    """Captured from the real game: every arrow must be certain, and the board
    must have a legal clear order (a deadlock here would mean a misread)."""
    image = cv2.imread(str(SAMPLES / name))
    assert image is not None, f"missing fixture {name}"
    config = Config().validate()
    detection = Detector(config).detect(image)

    assert len(detection.arrows) >= min_arrows, detection.describe()
    assert detection.scale >= scale_at_least
    low = [arrow for arrow in detection.arrows
           if arrow.confidence < config.min_direction_margin]
    assert low == [], f"{len(low)} ambiguous arrows: {low[:3]}"
    assert solver.order(detection.arrows) is not None
