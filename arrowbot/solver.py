"""Which arrows can leave the board without touching anything - pure geometry.

An arrow may only be tapped when nothing lies **in the corridor in front of its
head**: the half-plane it points into, restricted to a narrow band around its
own axis.  Removing an arrow can never put an obstacle into somebody else's
corridor, so the set of tappable arrows only ever grows and a greedy sweep is
provably optimal - if it gets stuck, no tap order could have cleared the board.

The functions here work on any object with ``head``, ``direction``, ``xs``,
``ys`` and ``scale`` attributes (see :class:`arrowbot.vision.Arrow`), so they can
be unit tested without OpenCV.
"""

from __future__ import annotations

from itertools import combinations, product
from typing import Dict, List, Optional, Sequence

import numpy as np

DIRECTIONS = ("U", "R", "D", "L")

#: half width of the corridor, in pixels at artwork scale 1.0 (the shaft is 4 px
#: wide and arrows on neighbouring rows are 27 px apart, so anything from about
#: 5 to 10 behaves identically)
CORRIDOR_HALF = 7.0


def blockers(arrow, alive: Sequence, half: Optional[float] = None,
             gap: float = 0.0) -> List:
    """Arrows whose ink sits in front of ``arrow`` (and therefore stop it)."""
    if half is None:
        half = CORRIDOR_HALF * float(getattr(arrow, "scale", 1.0))
    origin_x, origin_y = arrow.head
    direction = arrow.direction
    horizontal = direction in "LR"
    hit = []
    for other in alive:
        if other is arrow:
            continue
        if horizontal:
            band = (other.ys >= origin_y - half) & (other.ys <= origin_y + half)
            ahead = other.xs > origin_x + gap if direction == "R" else other.xs < origin_x - gap
        else:
            band = (other.xs >= origin_x - half) & (other.xs <= origin_x + half)
            ahead = other.ys > origin_y + gap if direction == "D" else other.ys < origin_y - gap
        if np.any(band & ahead):
            hit.append(other)
    return hit


def is_safe(arrow, alive: Sequence, half: Optional[float] = None,
            gap: float = 0.0) -> bool:
    """True when nothing is in the way of ``arrow``."""
    if half is None:
        half = CORRIDOR_HALF * float(getattr(arrow, "scale", 1.0))
    origin_x, origin_y = arrow.head
    direction = arrow.direction
    horizontal = direction in "LR"
    for other in alive:
        if other is arrow:
            continue
        if horizontal:
            band = (other.ys >= origin_y - half) & (other.ys <= origin_y + half)
            ahead = other.xs > origin_x + gap if direction == "R" else other.xs < origin_x - gap
        else:
            band = (other.xs >= origin_x - half) & (other.xs <= origin_x + half)
            ahead = other.ys > origin_y + gap if direction == "D" else other.ys < origin_y - gap
        if np.any(band & ahead):
            return False
    return True


def safe_arrows(arrows: Sequence, half: Optional[float] = None,
                gap: float = 0.0) -> List:
    """Every arrow that can be tapped right now."""
    return [arrow for arrow in arrows if is_safe(arrow, arrows, half, gap)]


def plan(arrows: Sequence, half: Optional[float] = None,
         gap: float = 0.0) -> Optional[List[List]]:
    """Greedy clear plan, as waves of arrows that can all go at once.

    Returns ``None`` when the board dead-locks: arrows remain but none of them
    can move, which is a contradiction only if the arrows were misread.
    """
    alive = list(arrows)
    waves: List[List] = []
    while alive:
        wave = safe_arrows(alive, half, gap)
        if not wave:
            return None
        waves.append(wave)
        alive = [arrow for arrow in alive if all(arrow is not gone for gone in wave)]
    return waves


def order(arrows: Sequence, half: Optional[float] = None,
          gap: float = 0.0) -> Optional[List]:
    """The same plan as one flat tap order."""
    waves = plan(arrows, half, gap)
    if waves is None:
        return None
    return [arrow for wave in waves for arrow in wave]


def clearable(arrows: Sequence, half: Optional[float] = None,
              gap: float = 0.0) -> bool:
    return plan(arrows, half, gap) is not None


class _DirectionView:
    """One arrow with a temporary direction, for repair search."""

    __slots__ = ("head", "direction", "xs", "ys", "scale", "confidence")

    def __init__(self, arrow, direction: str):
        self.head = arrow.head
        self.direction = direction
        self.xs = arrow.xs
        self.ys = arrow.ys
        self.scale = float(getattr(arrow, "scale", 1.0))
        self.confidence = float(getattr(arrow, "confidence", 1.0))


def repair(arrows: Sequence, half: Optional[float] = None, gap: float = 0.0,
           max_ambiguous: int = 6, max_changes: int = 2,
           max_checks: int = 120) -> Optional[Dict[object, str]]:
    """Find the smallest direction correction that makes a deadlocked board clear.

    A real Arrow Puzzle board is always solvable, so a mathematical deadlock on
    a detected board almost always means that one glyph's direction was misread.
    Every arrow carries the template scores for all four directions; this
    searches the few most ambiguous arrows, cheapest correction first, and
    returns ``{arrow: direction}`` for the first assignment that clears.

    ``None`` means no small correction helps (the board really is blocked, or
    something that is not an arrow was picked up).  The caller should re-read
    the frame before concluding anything.
    """
    arrows = list(arrows)
    if not arrows:
        return {}
    if clearable(arrows, half, gap):
        return {}

    ranked = sorted(arrows, key=lambda arrow: float(getattr(arrow, "confidence", 1.0)))
    candidates = ranked[: max(0, int(max_ambiguous))]
    options = []
    for arrow in candidates:
        scores = getattr(arrow, "scores", None) or {}
        alternatives = [direction for direction in DIRECTIONS
                        if direction != arrow.direction]
        if scores:
            best = max(scores.values())
            alternatives.sort(key=lambda direction: scores.get(direction, -1.0),
                              reverse=True)
            # an alternative more than half a score behind the winner is not
            # credible; keeping it would let the search "solve" the board by
            # inventing an arrow that is not there
            alternatives = [direction for direction in alternatives
                            if scores.get(direction, -1.0) >= best - 0.5]
        options.append((arrow, alternatives))
    options = [(arrow, dirs) for arrow, dirs in options if dirs]
    if not options:
        return None

    checks = 0
    for changes in range(1, min(len(options), int(max_changes)) + 1):
        for chosen in combinations(range(len(options)), changes):
            for directions in product(*(options[index][1] for index in chosen)):
                fixes = {options[index][0]: direction
                         for index, direction in zip(chosen, directions)}
                views = [_DirectionView(arrow, fixes.get(arrow, arrow.direction))
                         for arrow in arrows]
                checks += 1
                if clearable(views, half, gap):
                    return fixes
                if checks >= max_checks:
                    return None
    return None
