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

from typing import List, Optional, Sequence

import numpy as np

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
