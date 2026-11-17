"""A synthetic phone used by the self-test (and for the sample screenshot).

``SimGame`` offers the same two methods as the real ADB device (``screencap`` and
``tap``) and draws boards the way the game does: dense, bent, axis-aligned arrows
with one solid triangular head each, on a light board with a HUD above it.

Crucially it keeps the ground truth as *pixel* arrows and applies exactly the
same corridor rule as the solver, so a tap the agent thinks is safe is safe in
the simulation too.  Boards are generated in the reverse order of a legal clear
(insert an arrow only where its corridor is still empty), which guarantees that
they can be cleared.

It also mimics two details that matter on the real device:

* **the camera zooms** as a board empties, so the artwork scale grows,
* **tapping empty paper loses a heart**, and a heart loss is visible in the HUD.
"""

from __future__ import annotations

import random
import time
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from . import solver
from .config import Config
from .vision import HEAD_APEX_OFFSET, HEAD_BASE, HEAD_HEIGHT, VECTORS

#: stroke width of an arrow shaft, in pixels at artwork scale 1.0
STROKE = 4.0
#: a real board packs arrows on a 27 px lattice
PITCH = 27.0
#: longest arrow, in cells
MAX_CELLS = 6


class SimArrow:
    """Ground truth for one arrow, in board coordinates at artwork scale 1."""

    __slots__ = ("path", "head", "direction", "xs", "ys", "scale")

    def __init__(self, path: List[Tuple[float, float]], direction: str,
                 xs: np.ndarray, ys: np.ndarray):
        self.path = path
        self.head = (int(round(path[-1][0])), int(round(path[-1][1])))
        self.direction = direction
        self.xs = xs
        self.ys = ys
        self.scale = 1.0

    def __str__(self) -> str:
        return f"{self.direction}@({self.path[-1][0]:.0f},{self.path[-1][1]:.0f})"


def arrow_mask(points: Sequence[Tuple[float, float]], head: Tuple[float, float],
               direction: str, pad: int = 6) -> Tuple[np.ndarray, int, int]:
    """Rasterise one arrow the way the game draws it (triangle + shaft)."""
    vx, vy = VECTORS[direction]
    apex = (head[0] + vx * HEAD_APEX_OFFSET, head[1] + vy * HEAD_APEX_OFFSET)
    base = (apex[0] - vx * HEAD_HEIGHT, apex[1] - vy * HEAD_HEIGHT)
    half = HEAD_BASE / 2.0
    left = (base[0] - vy * half, base[1] + vx * half)
    right = (base[0] + vy * half, base[1] - vx * half)

    shaft = [(float(x), float(y)) for x, y in points]
    shaft[-1] = base
    corners = np.array([apex, base, left, right] + list(points), np.float64)
    x0 = int(np.floor(corners[:, 0].min())) - pad
    y0 = int(np.floor(corners[:, 1].min())) - pad
    x1 = int(np.ceil(corners[:, 0].max())) + pad
    y1 = int(np.ceil(corners[:, 1].max())) + pad

    canvas = np.zeros((y1 - y0 + 1, x1 - x0 + 1), np.uint8)
    offset = np.array([-x0, -y0], np.float64)
    line = np.round(np.array(shaft, np.float64) + offset).astype(np.int32)
    cv2.polylines(canvas, [line], False, 1, int(STROKE), cv2.LINE_8)
    triangle = np.round(np.array([apex, left, right], np.float64) + offset).astype(np.int32)
    cv2.fillConvexPoly(canvas, triangle, 1)
    rows, columns = np.nonzero(canvas)
    return columns + x0, rows + y0, x0


class SimGame:
    """Renders a board, accepts taps, and loses hearts for wrong taps."""

    def __init__(
        self,
        config: Optional[Config] = None,
        cols: int = 13,
        rows: int = 15,
        seed: int = 1,
        width: int = 720,
        height: int = 1560,
        background: Tuple[int, int, int] = (250, 250, 250),
        foreground: Tuple[int, int, int] = (48, 60, 96),
        advance_delay: float = 0.15,
    ):
        self.config = config or Config()
        self.cols, self.rows = cols, rows
        self.width, self.height = width, height
        self.background = background
        self.panel = tuple(max(0, value - 8) for value in background)
        self.foreground = foreground
        self.advance_delay = advance_delay
        self.rng = random.Random(seed)

        self.board_px = (cols * PITCH, rows * PITCH)
        self.lives = 3
        self.lives_lost = 0
        self.wins = 0
        self.level = 1
        #: persistent camera zoom on top of the automatic one (1.0 = fitted)
        self.camera_zoom = 1.0
        self.board: List[SimArrow] = []
        self._level_size = 1
        self._cleared_at: Optional[float] = None
        self._new_level()

    # ------------------------------------------------------------------ setup
    def reshape(self, cols: int, rows: int) -> None:
        """Change the board size, like a later level of the real game."""
        self.cols, self.rows = cols, rows
        self.board_px = (cols * PITCH, rows * PITCH)
        self._new_level()

    def reset(self) -> None:
        self.lives = 3
        self._new_level()

    def _new_level(self) -> None:
        self.board = self._generate()
        self._level_size = max(1, len(self.board))
        self._cleared_at = None

    def _cell_centre(self, cell: Tuple[int, int]) -> Tuple[float, float]:
        return ((cell[1] + 0.5) * PITCH, (cell[0] + 0.5) * PITCH)

    def _free(self, cell: Tuple[int, int], step: Tuple[int, int],
              used: np.ndarray, taken: set) -> bool:
        row, col = cell[0] + step[0], cell[1] + step[1]
        return (0 <= row < self.rows and 0 <= col < self.cols
                and not used[row, col] and (row, col) not in taken)

    def _grow(self, start: Tuple[int, int],
              used: np.ndarray) -> Optional[List[Tuple[int, int]]]:
        """Random bent path of 3..6 free cells starting at ``start``.

        Cells of this arrow's own path are reserved as it grows, so an arrow can
        never double back over itself (that would merge two of its strokes and
        make the head unreadable).
        """
        path = [start]
        taken = {start}
        heading: Optional[Tuple[int, int]] = None
        target = self.rng.randint(3, MAX_CELLS)
        for _ in range(MAX_CELLS * 4):
            if len(path) >= target:
                break
            if heading is None or self.rng.random() < 0.35:
                options = [step for step in ((-1, 0), (1, 0), (0, -1), (0, 1))
                           if self._free(path[-1], step, used, taken)]
                if not options:
                    break
                heading = self.rng.choice(options)
            if not self._free(path[-1], heading, used, taken):
                heading = None
                continue
            path.append((path[-1][0] + heading[0], path[-1][1] + heading[1]))
            taken.add(path[-1])
        return path if len(path) >= 3 else None

    def _make_arrow(self, path: List[Tuple[int, int]]) -> Optional[SimArrow]:
        points = [self._cell_centre(cell) for cell in path]
        direction = self._direction_of(points[-2], points[-1])
        if direction is None:
            return None
        xs, ys, _origin = arrow_mask(points, points[-1], direction)
        return SimArrow(points, direction, xs, ys)

    @staticmethod
    def _direction_of(back: Sequence[float], front: Sequence[float]) -> Optional[str]:
        dx, dy = front[0] - back[0], front[1] - back[1]
        if abs(dx) >= abs(dy) and dx:
            return "R" if dx > 0 else "L"
        if dy:
            return "D" if dy > 0 else "U"
        return None

    def _generate(self) -> List[SimArrow]:
        used = np.zeros((self.rows, self.cols), bool)
        cells = [(row, col) for row in range(self.rows) for col in range(self.cols)]
        self.rng.shuffle(cells)
        board: List[SimArrow] = []
        for start in cells:
            if used[start]:
                continue
            for _attempt in range(3):
                path = self._grow(start, used)
                if path is None:
                    break
                arrow = self._make_arrow(path)
                if arrow is None:
                    break
                if solver.blockers(arrow, board, half=solver.CORRIDOR_HALF):
                    continue          # would be born blocked -> not solvable this way
                for cell in path:
                    used[cell] = True
                board.append(arrow)
                break
        return board

    # --------------------------------------------------------------- geometry
    @property
    def zoom(self) -> float:
        """The camera zooms in as the board empties, like the real game."""
        remaining = len(self.board) / self._level_size if self._level_size else 1.0
        return 1.0 + 0.30 * (1.0 - remaining)

    def _transform(self) -> Tuple[float, float, float]:
        zoom = self.zoom * self.camera_zoom
        board_width, board_height = self.board_px
        offset_x = self.width / 2.0 - zoom * board_width / 2.0
        offset_y = self.height / 2.0 - zoom * board_height / 2.0
        return zoom, offset_x, offset_y

    def to_screen(self, x: float, y: float) -> Tuple[float, float]:
        zoom, offset_x, offset_y = self._transform()
        return offset_x + zoom * x, offset_y + zoom * y

    def to_board(self, x: float, y: float) -> Tuple[float, float]:
        zoom, offset_x, offset_y = self._transform()
        return (x - offset_x) / zoom, (y - offset_y) / zoom

    # ------------------------------------------------------------- rendering
    def screencap(self) -> np.ndarray:
        if not self.board:
            self._maybe_advance()
        image = np.empty((self.height, self.width, 3), np.uint8)
        image[:] = self.background
        self._draw_hud(image)

        for arrow in self.board:
            self._draw_arrow(image, arrow)
        if not self.board:
            self._draw_win_panel(image)
        return image

    def _draw_hud(self, image: np.ndarray) -> None:
        cv2.putText(image, f"LEVEL {self.level}", (24, 64),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, self.foreground, 2, cv2.LINE_AA)
        x0, y0, x1, y1 = (int(round(v * (self.width if index % 2 == 0 else self.height)))
                          for index, v in enumerate(self.config.hearts_roi))
        step = (x1 - x0) / 3.0
        radius = max(6, int(round((y1 - y0) * 0.36)))
        for index in range(self.lives):
            centre = (int(x0 + step * (index + 0.5)), int((y0 + y1) / 2))
            cv2.circle(image, centre, radius, (40, 40, 220), -1, cv2.LINE_AA)

    def _draw_arrow(self, image: np.ndarray, arrow: SimArrow) -> None:
        zoom, _ox, _oy = self._transform()
        points = [self.to_screen(x, y) for x, y in arrow.path]
        vx, vy = VECTORS[arrow.direction]
        apex = self.to_screen(arrow.path[-1][0] + vx * HEAD_APEX_OFFSET,
                              arrow.path[-1][1] + vy * HEAD_APEX_OFFSET)
        base_x = apex[0] - vx * HEAD_HEIGHT * zoom
        base_y = apex[1] - vy * HEAD_HEIGHT * zoom
        half = HEAD_BASE * zoom / 2.0
        left = (base_x - vy * half, base_y + vx * half)
        right = (base_x + vy * half, base_y - vx * half)

        shaft = [np.round(point).astype(np.int32) for point in points]
        shaft[-1] = (int(round(base_x)), int(round(base_y)))
        cv2.polylines(image, [np.array(shaft, np.int32)], False,
                      self.foreground, max(1, int(round(STROKE * zoom))), cv2.LINE_8)
        triangle = np.round(np.array([apex, left, right], np.float64)).astype(np.int32)
        cv2.fillConvexPoly(image, triangle, self.foreground)

    def _draw_win_panel(self, image: np.ndarray) -> None:
        top = int(self.height * 0.42)
        cv2.rectangle(image, (60, top), (self.width - 60, top + 90), (60, 150, 90), -1)
        cv2.putText(image, "LEVEL CLEAR", (90, top + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3, cv2.LINE_AA)

    # ---------------------------------------------------------------- input
    def pinch(self, center_x: float, center_y: float,
              start_gap: float, end_gap: float,
              steps: int = 8, duration_ms: float = 200.0, **kwargs) -> None:
        """Two-finger pinch; fingers together zoom the camera out, as expected."""
        if start_gap <= 0:
            return
        factor = float(end_gap) / float(start_gap)
        self.camera_zoom = float(min(4.0, max(1.0, self.camera_zoom * factor)))

    def tap_many(self, points, gap_ms: float = 0.0, hold_ms: float = 0.0) -> int:
        for x, y in points:
            self.tap(x, y)
        return len(points)

    def tap(self, x: int, y: int) -> None:
        if not self.board:
            return
        board_x, board_y = self.to_board(x, y)
        target = None
        best = 1e9
        for arrow in self.board:
            distance = float(np.min((arrow.xs - board_x) ** 2 + (arrow.ys - board_y) ** 2))
            if distance < best:
                best, target = distance, arrow
        if target is None or best > 12.0 ** 2:
            self._lose_life()          # tapped empty paper
            return
        if solver.is_safe(target, self.board, half=solver.CORRIDOR_HALF):
            self.board = [arrow for arrow in self.board if arrow is not target]
            if not self.board:
                self.wins += 1
                self._cleared_at = time.monotonic()
        else:
            self._lose_life()

    def _lose_life(self) -> None:
        self.lives_lost += 1
        self.lives -= 1
        if self.lives <= 0:
            self.reset()

    def _maybe_advance(self) -> None:
        if self._cleared_at is None:
            self._cleared_at = time.monotonic()
        if time.monotonic() - self._cleared_at < self.advance_delay:
            return
        self.level += 1
        self.reset()
        self._cleared_at = None

    # ------------------------------------------------------------- reporting
    def truth(self) -> Dict[Tuple[int, int], str]:
        """Ground truth as ``{(row, col): direction}`` for the head cells."""
        result: Dict[Tuple[int, int], str] = {}
        for arrow in self.board:
            x, y = arrow.path[-1]
            result[(int(y // PITCH), int(x // PITCH))] = arrow.direction
        return result
