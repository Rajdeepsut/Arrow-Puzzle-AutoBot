"""Pure-logic ray-casting solver for Arrow Puzzle boards.

A board is a mapping ``(row, col) -> direction`` with direction in ``UDLR``.
Row 0 is the top row and column 0 the left column, so ``U`` means "forward is
-row" (screen coordinates, y grows downwards).

Why greedily tapping every safe arrow is always correct
------------------------------------------------------
An arrow is *safe* when no other arrow sits strictly in front of it along its
own ray.  Tapping a safe arrow only ever removes an arrow from the board, and
removing an arrow can never turn a safe arrow into a blocked one.  The set of
safe arrows therefore only grows as the board empties, which means:

* the order in which safe arrows are tapped cannot hurt, and
* if the greedy sweep gets stuck with arrows left over, *no* tap order could
  have cleared that position.  It is dead-locked, not unlucky.

That single property is what makes this bot deterministic and cheap: one sweep
per wave, no search, no backtracking.

This module deliberately has no third-party imports so it can be tested and
reasoned about in isolation.
"""

from __future__ import annotations

import random
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

Cell = Tuple[int, int]
Board = Dict[Cell, str]

#: forward step per direction, in (row, col) space
DELTAS: Dict[str, Cell] = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}
DIRECTIONS: Tuple[str, ...] = ("U", "D", "L", "R")
OPPOSITE: Dict[str, str] = {"U": "D", "D": "U", "L": "R", "R": "L"}
#: convenient aliases used by the text board parser
GLYPHS: Dict[str, str] = {"^": "U", "v": "D", "<": "L", ">": "R"}


# --------------------------------------------------------------------------- #
# core rules
# --------------------------------------------------------------------------- #
def bounds(cells: Mapping[Cell, str]) -> Tuple[int, int, int, int]:
    """``(row_min, row_max, col_min, col_max)`` of the occupied cells."""
    if not cells:
        return (0, -1, 0, -1)
    rows = [cell[0] for cell in cells]
    cols = [cell[1] for cell in cells]
    return (min(rows), max(rows), min(cols), max(cols))


def blockers(cells: Mapping[Cell, str], cell: Cell) -> List[Cell]:
    """Return the cells strictly in front of ``cell``, nearest first.

    The ray travels *through* empty cells and stops at the edge of the board.
    The edge is derived from the arrows themselves: every row/column outside the
    occupied bounding box is empty by definition, so it behaves exactly like the
    real border for the purpose of this test.
    """
    direction = cells.get(cell)
    if direction not in DELTAS:
        return []
    row_min, row_max, col_min, col_max = bounds(cells)
    dr, dc = DELTAS[direction]
    r, c = cell[0] + dr, cell[1] + dc
    found: List[Cell] = []
    while row_min <= r <= row_max and col_min <= c <= col_max:
        if (r, c) in cells:
            found.append((r, c))
        r += dr
        c += dc
    return found


def is_safe(cells: Mapping[Cell, str], cell: Cell) -> bool:
    """True when tapping ``cell`` cannot collide with another arrow.

    Unknown directions (``"?"``, i.e. vision could not classify the glyph)
    are never considered safe - we do not gamble with lives.
    """
    if cells.get(cell) not in DELTAS:
        return False
    return not blockers(cells, cell)


def safe_cells(cells: Mapping[Cell, str]) -> List[Cell]:
    """Every currently safe arrow, sorted for reproducible logs."""
    return sorted(c for c in cells if is_safe(cells, c))


def plan(cells: Mapping[Cell, str], limit: int = 1000) -> Optional[List[List[Cell]]]:
    """Solve the whole board as a list of waves (``None`` when dead-locked).

    A wave is a set of arrows that can all be tapped back to back without
    re-reading the screen, because tapping a safe arrow never endangers
    another safe arrow.
    """
    work: Board = dict(cells)
    waves: List[List[Cell]] = []
    while work:
        if len(waves) >= limit:
            return None
        wave = safe_cells(work)
        if not wave:
            return None
        waves.append(wave)
        for cell in wave:
            work.pop(cell, None)
    return waves


def simulate(cells: Mapping[Cell, str], waves: Sequence[Sequence[Cell]]) -> bool:
    """Return True when ``waves`` legally clears ``cells`` (used by tests)."""
    work: Board = dict(cells)
    for wave in waves:
        for cell in wave:
            if not is_safe(work, cell):
                return False
            work.pop(cell, None)
    return not work


# --------------------------------------------------------------------------- #
# text helpers (debugging, tests, ``main.py solve``)
# --------------------------------------------------------------------------- #
def parse_ascii(text: str) -> Board:
    """Parse a text board: one row per line, ``.`` for an empty cell.

    ``U D L R`` (case insensitive) and ``^ v < >`` are accepted, both as a
    compact string (``"U.R"``) and whitespace separated (``"U . R"``).
    """
    board: Board = {}
    lines = [ln for ln in text.replace("|", "").splitlines() if ln.strip()]
    for r, line in enumerate(lines):
        tokens = line.split()
        if len(tokens) == 1:
            tokens = list(tokens[0])
        for c, token in enumerate(tokens):
            token = token.strip()
            if not token:
                continue
            if token in GLYPHS:
                board[(r, c)] = GLYPHS[token]
            elif token.upper() in DELTAS:
                board[(r, c)] = token.upper()
    return board


def format_ascii(cells: Mapping[Cell, str]) -> str:
    """Render a board as an aligned text grid."""
    if not cells:
        return "(empty)"
    rmin = min(r for r, _ in cells)
    rmax = max(r for r, _ in cells)
    cmin = min(c for _, c in cells)
    cmax = max(c for _, c in cells)
    return "\n".join(
        " ".join(cells.get((r, c), ".") for c in range(cmin, cmax + 1))
        for r in range(rmin, rmax + 1)
    )


# --------------------------------------------------------------------------- #
# board generation (self-test / simulation only)
# --------------------------------------------------------------------------- #
def random_solvable_board(
    rows: int, cols: int, rng: random.Random, fill: float = 1.0
) -> Board:
    """Build a board that is guaranteed solvable by construction.

    Cells are visited in *reverse* clearing order.  A cell receives a direction
    whose forward ray contains no already-placed arrow, so when its own turn
    comes (in forward order) everything in front of it has already been removed.
    A cell that is fully surrounded in all four directions is simply left empty -
    dropping an arrow can only make the board easier, so the result stays
    solvable while still looking like a real, dense level.
    """
    cells: List[Cell] = [(r, c) for r in range(rows) for c in range(cols)]
    order = cells[:]
    rng.shuffle(order)
    board: Board = {}
    for cell in reversed(order):
        r, c = cell
        options: List[str] = []
        for direction, (dr, dc) in DELTAS.items():
            rr, cc = r + dr, c + dc
            blocked = False
            while 0 <= rr < rows and 0 <= cc < cols:
                if (rr, cc) in board:
                    blocked = True
                    break
                rr += dr
                cc += dc
            if not blocked:
                options.append(direction)
        if options and (fill >= 1.0 or rng.random() < fill):
            board[cell] = rng.choice(options)
    return board
