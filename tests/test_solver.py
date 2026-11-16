"""Unit tests for the pure solver (no OpenCV, no phone).

    pytest -q
"""

from __future__ import annotations

import random

import pytest

from arrowbot import solver


def test_open_lane_is_safe():
    board = solver.parse_ascii("U...")
    assert solver.is_safe(board, (0, 0))
    assert solver.blockers(board, (0, 0)) == []


def test_empty_cells_do_not_hide_a_blocker():
    """Regression: rays must travel *through* empty cells, not stop at them."""
    board = solver.parse_ascii(
        """
        D
        .
        U
        """
    )
    assert board == {(0, 0): "D", (2, 0): "U"}
    assert solver.blockers(board, (0, 0)) == [(2, 0)]
    assert not solver.is_safe(board, (0, 0))
    assert solver.plan(board) is None


def test_only_the_arrow_in_front_blocks_the_lane():
    board = solver.parse_ascii("L.L")  # both point left
    assert solver.blockers(board, (0, 0)) == []  # nothing left of column 0
    assert solver.blockers(board, (0, 2)) == [(0, 0)]
    assert solver.safe_cells(board) == [(0, 0)]


def test_chain_clears_one_arrow_at_a_time():
    board = solver.parse_ascii(
        """
        U
        U
        U
        U
        """
    )
    waves = solver.plan(board)
    assert waves == [[(0, 0)], [(1, 0)], [(2, 0)], [(3, 0)]]
    assert solver.simulate(board, waves)


def test_wave_can_tap_everything_at_once():
    board = solver.parse_ascii(
        """
        U U U
        . . .
        D D D
        """
    )
    waves = solver.plan(board)
    assert waves is not None and len(waves) == 1
    assert len(waves[0]) == 6


def test_perpendicular_arrows_do_not_block_each_other():
    board = solver.parse_ascii(
        """
        R . .
        . . .
        . . U
        """
    )
    assert set(solver.safe_cells(board)) == {(0, 0), (2, 2)}


def test_unknown_direction_is_never_tappable():
    board = {(0, 0): "?", (0, 1): "U"}
    assert not solver.is_safe(board, (0, 0))
    assert solver.safe_cells(board) == [(0, 1)]


def test_text_round_trip():
    text = "U . R\n. D .\nL . R"
    board = solver.parse_ascii(text)
    assert len(board) == 5
    assert solver.format_ascii(board) == text


def test_glyph_aliases():
    board = solver.parse_ascii("^>v<")
    assert list(board.values()) == ["U", "R", "D", "L"]


@pytest.mark.parametrize("seed", range(20))
def test_random_boards_are_planned_legally(seed: int):
    rng = random.Random(seed)
    rows, cols = rng.randint(3, 9), rng.randint(3, 9)
    board = solver.random_solvable_board(rows, cols, rng)
    waves = solver.plan(board)
    assert waves is not None, f"generator produced an unsolvable {rows}x{cols} board"
    assert solver.simulate(board, waves)
    assert all(wave for wave in waves)
    assert sum(len(wave) for wave in waves) == len(board)


def test_plan_never_suggests_a_blocked_tap():
    rng = random.Random(7)
    for _ in range(50):
        board = solver.random_solvable_board(6, 6, rng)
        work = dict(board)
        for cell in solver.safe_cells(work):
            assert solver.is_safe(work, cell)
            work.pop(cell)
