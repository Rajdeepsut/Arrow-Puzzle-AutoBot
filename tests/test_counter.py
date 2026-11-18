"""Tests for the HUD remaining-arrow counter (real captures, no phone)."""

from __future__ import annotations

import pathlib

import cv2
import numpy as np
import pytest

from arrowbot.counter import counter_glyphs, read_counter

SAMPLES = pathlib.Path(__file__).resolve().parents[1] / "samples"


@pytest.mark.parametrize("name,expected", [
    ("real_level.png", 56),
    ("real_level_74.png", 74),
    ("real_level_zoomed.png", 78),
])
def test_real_counters_are_read_exactly(name, expected):
    image = cv2.imread(str(SAMPLES / name))
    assert image is not None, f"missing fixture {name}"
    assert read_counter(image) == expected


@pytest.mark.parametrize("scale", [0.75, 1.2])
def test_the_counter_survives_a_different_screen_size(scale):
    image = cv2.imread(str(SAMPLES / "real_level_74.png"))
    resized = cv2.resize(image, None, fx=scale, fy=scale,
                         interpolation=cv2.INTER_AREA)
    assert read_counter(resized) == 74


def test_a_blank_hud_is_unreadable_not_guessed():
    blank = np.full((2392, 1080, 3), 255, np.uint8)
    assert read_counter(blank) is None
    assert counter_glyphs(blank) == []


def test_the_wide_arrow_icon_is_not_mistaken_for_a_digit():
    image = cv2.imread(str(SAMPLES / "real_level_74.png"))
    glyphs = counter_glyphs(image)
    assert len(glyphs) == 2                 # the digits 7 and 4 only
    assert all(glyph.shape == (32, 20) for glyph in glyphs)


def test_counter_defaults_are_sane():
    from arrowbot.config import Config

    config = Config().validate()
    assert config.counter_check is True
    assert len(config.counter_roi) == 4
    assert 0.0 <= config.counter_min_score <= 1.0


def test_a_bad_roi_is_normalised():
    from arrowbot.config import Config

    config = Config().validate()
    config.counter_roi = [-1.0, 0.05, 2.0, 0.15]
    config.validate()
    assert config.counter_roi == [0.0, 0.05, 1.0, 0.15]
