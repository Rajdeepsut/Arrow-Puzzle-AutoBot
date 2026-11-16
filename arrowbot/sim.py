"""A synthetic phone used by the self-test.

``SimGame`` implements the same two methods as the real ADB device
(``screencap`` and ``tap``) and draws its boards the way the game does: dark
arrow glyphs on a light panel, a HUD line above the board and a level-complete
screen with a button.  That makes it possible to exercise the *whole* pipeline
(vision -> solver -> taps -> next level) without a phone.
"""

from __future__ import annotations

import random
import time
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from . import solver
from .config import Config

#: arrow glyph pointing right, in cell-relative coordinates
ARROW_SHAPE = np.array(
    [
        [-0.46, -0.15],
        [0.02, -0.15],
        [0.02, -0.33],
        [0.48, 0.00],
        [0.02, 0.33],
        [0.02, 0.15],
        [-0.46, 0.15],
    ],
    dtype=np.float64,
)

_QUARTER_TURNS = {"R": 0, "D": 1, "L": 2, "U": 3}


def arrow_polygon(direction: str, size: float) -> np.ndarray:
    """Rotate the base glyph so that it points in ``direction``."""
    points = ARROW_SHAPE * size
    for _ in range(_QUARTER_TURNS[direction]):
        points = np.stack([-points[:, 1], points[:, 0]], axis=1)
    return points


class SimGame:
    """Renders a board, accepts taps, and can lose lives."""

    def __init__(
        self,
        config: Optional[Config] = None,
        rows: int = 6,
        cols: int = 6,
        seed: int = 1,
        width: int = 720,
        height: int = 1280,
        background: Tuple[int, int, int] = (248, 248, 248),
        foreground: Tuple[int, int, int] = (45, 45, 45),
    ):
        self.config = config or Config()
        self.rows, self.cols = rows, cols
        self.width, self.height = width, height
        self.background = background
        self.foreground = foreground
        # a slightly raised panel, like the real game's board area
        self.panel = tuple(
            min(255, value + 14) if sum(background) < 384 else max(0, value - 16)
            for value in background
        )
        self.rng = random.Random(seed)

        self.cell = int(min((width - 120) / cols, (height - 520) / rows))
        self.board_w = cols * self.cell
        self.board_h = rows * self.cell
        self.x0 = (width - self.board_w) // 2
        self.y0 = (height - self.board_h) // 2

        self.lives = 3
        self.lives_lost = 0
        self.wins = 0
        self.level = 1
        self.board: Dict[Tuple[int, int], str] = {}
        self._win_at = 0.0
        self._new_level()

    def reshape(self, rows: int, cols: int) -> None:
        """Change the grid size, like a later level of the real game."""
        self.rows, self.cols = rows, cols
        self.cell = int(min((self.width - 120) / cols, (self.height - 520) / rows))
        self.board_w = cols * self.cell
        self.board_h = rows * self.cell
        self.x0 = (self.width - self.board_w) // 2
        self.y0 = (self.height - self.board_h) // 2
        self._new_level()

    # ------------------------------------------------------------------ setup
    def _new_level(self) -> None:
        self.board = solver.random_solvable_board(self.rows, self.cols, self.rng)

    def reset(self) -> None:
        self.lives = 3
        self.board = {}
        self._new_level()

    # ------------------------------------------------------------- geometry
    def cell_centre(self, row: int, col: int) -> Tuple[int, int]:
        return (
            int(self.x0 + self.cell * (col + 0.5)),
            int(self.y0 + self.cell * (row + 0.5)),
        )

    def cell_at(self, x: int, y: int) -> Optional[Tuple[int, int]]:
        col = int((x - self.x0) // self.cell)
        row = int((y - self.y0) // self.cell)
        if 0 <= row < self.rows and 0 <= col < self.cols:
            return (row, col)
        return None

    # ------------------------------------------------------------- rendering
    def screencap(self) -> np.ndarray:
        image = np.empty((self.height, self.width, 3), np.uint8)
        image[:] = self.background

        # board panel (large, so the blob filter ignores it)
        cv2.rectangle(
            image,
            (self.x0 - 16, self.y0 - 16),
            (self.x0 + self.board_w + 16, self.y0 + self.board_h + 16),
            self.panel,
            -1,
        )
        cv2.putText(
            image, f"LEVEL {self.level}", (40, 96),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, self.foreground, 2, cv2.LINE_AA,
        )
        cv2.putText(
            image, "Easybrain", (40, 1232),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 2, cv2.LINE_AA,
        )

        if not self.board:
            self._draw_win_screen(image)
            return image

        for (row, col), direction in self.board.items():
            cx, cy = self.cell_centre(row, col)
            polygon = arrow_polygon(direction, self.cell * 0.98)
            polygon = np.round(polygon + np.array([cx, cy])).astype(np.int32)
            cv2.fillPoly(image, [polygon], self.foreground, cv2.LINE_AA)
        return image

    def _draw_win_screen(self, image: np.ndarray) -> None:
        cv2.rectangle(
            image,
            (self.x0 - 16, self.y0 - 16),
            (self.x0 + self.board_w + 16, self.y0 + self.board_h + 16),
            self.panel,
            -1,
        )
        cv2.rectangle(image, (110, 980), (610, 1120), (40, 150, 90), -1)
        cv2.putText(
            image, "NEXT", (300, 1070),
            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3, cv2.LINE_AA,
        )

    # ---------------------------------------------------------------- input
    def tap(self, x: int, y: int) -> None:
        point = self.cell_at(x, y)
        if point is None:
            if not self.board:
                self._advance()
            return
        if point not in self.board:
            return
        if solver.is_safe(self.board, point):
            del self.board[point]
            if not self.board:
                self.wins += 1
                self._win_at = time.monotonic()
        else:
            self.lives_lost += 1
            self.lives -= 1
            if self.lives <= 0:
                self.reset()

    def _advance(self) -> None:
        # a real game ignores taps until the win animation has played
        if time.monotonic() - self._win_at < 0.2:
            return
        self.level += 1
        self.board = {}
        self._new_level()
