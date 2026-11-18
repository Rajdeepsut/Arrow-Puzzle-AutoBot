"""Reading an Arrow Puzzle board out of a phone screenshot.

The game draws every arrow as one solid, bent, axis-aligned stroke with a thick
triangular head at one end.  That gives a very direct way to read a board:

1. **ink** - arrows are navy (``grey < 110``) or red; everything else is ignored,
2. **heads** - the head is the only thick ink on the board, so it is a maximum of
   the distance transform (dt ~ 5.0 on a full board vs ~2.2 on a shaft),
3. **one arrow per head** - touching arrows share one connected blob, so blobs
   are split between their heads (geodesic Voronoi),
4. **direction** - the 27x27 patch around a head is matched against a synthetic
   triangle template and its three rotations,
5. **scale** - the camera zooms as a board empties (dt 5.0 -> 8.0).  Since the
   head is the thickest ink, ``dt.max() / 5.0`` is the artwork scale, and the
   template, the thickness threshold and the corridor all scale with it.

No grid, no calibration image, no fixed cell size.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

DIRECTIONS = ("U", "R", "D", "L")
VECTORS: Dict[str, Tuple[int, int]] = {"U": (0, -1), "D": (0, 1), "L": (-1, 0), "R": (1, 0)}

#: dt at the centre of a real arrow head, on the reference artwork (scale 1.0)
HEAD_DT = 5.0
#: head geometry, measured on real screenshots: apex 8 px in front of the head
#: centre, 11 px tall, 15 px wide at the base, with a 3 px shaft stub behind it
HEAD_APEX_OFFSET = 8.0
HEAD_HEIGHT = 11.0
HEAD_BASE = 15.0
SHAFT_HALF_WIDTH = 1.5
SHAFT_STUB = 10.0
#: half-size of the 27x27 patch a head is matched in
PATCH_RADIUS = 13.0

Box = Tuple[int, int, int, int]


# --------------------------------------------------------------------------- #
# ink
# --------------------------------------------------------------------------- #
def ink_mask(
    image: np.ndarray,
    roi: Optional[Sequence[float]] = None,
    exclude_rects: Optional[Sequence[Sequence[float]]] = None,
    gray_max: int = 110,
    red: bool = True,
) -> np.ndarray:
    """Binary mask of arrow ink, restricted to ``roi`` (screen fractions).

    Navy is any sufficiently dark pixel; red arrows are found by their colour
    signature.  Both kinds are ordinary arrows and are treated identically.
    """
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    blue, green, red_channel = cv2.split(image.astype(np.int16))
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = grey < gray_max
    if red:
        mask |= (
            (red_channel > 130)
            & (green < 130)
            & (blue < 130)
            & (red_channel - green > 40)
            & (red_channel - blue > 40)
        )

    height, width = mask.shape
    x0, y0, x1, y1 = fractions_to_box(roi, width, height)
    keep = np.zeros_like(mask)
    keep[y0:y1, x0:x1] = True
    for rect in exclude_rects or ():
        rx0, ry0, rx1, ry1 = fractions_to_box(rect, width, height)
        keep[ry0:ry1, rx0:rx1] = False
    return (mask & keep).astype(np.uint8)


def fractions_to_box(
    fractions: Optional[Sequence[float]], width: int, height: int
) -> Box:
    """``(x0, y0, x1, y1)`` in pixels from screen fractions (default: everything)."""
    if not fractions or len(fractions) != 4:
        return 0, 0, width, height
    x0, y0, x1, y1 = fractions
    return (
        int(round(x0 * width)),
        int(round(y0 * height)),
        int(round(x1 * width)),
        int(round(y1 * height)),
    )


def bbox_of(mask: np.ndarray) -> Box:
    rows, columns = np.nonzero(mask)
    if len(columns) == 0:
        return 0, 0, 0, 0
    return (int(columns.min()), int(rows.min()), int(columns.max()), int(rows.max()))


# --------------------------------------------------------------------------- #
# scale
# --------------------------------------------------------------------------- #
def estimate_scale(mask: np.ndarray, dt: Optional[np.ndarray] = None,
                   limits: Sequence[float] = (0.5, 4.0)) -> float:
    """Artwork scale: the head is the thickest ink, so ``dt.max() / 5``.

    The game zooms the camera in as a board empties, which grows every part of
    the drawing; this one number tells the rest of the pipeline how big things
    are *right now*.
    """
    if dt is None:
        dt = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    if not dt.any():
        return 1.0
    scale = float(dt.max()) / HEAD_DT
    low, high = float(limits[0]), float(limits[1])
    return min(max(scale, low), high)


# --------------------------------------------------------------------------- #
# heads
# --------------------------------------------------------------------------- #
def component_pixels(labels: np.ndarray, stats: np.ndarray, label: int
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """Pixels of one labelled component, without scanning the whole image."""
    left, top, width, height = (int(v) for v in stats[label, :4])
    window = labels[top:top + height, left:left + width] == label
    rows, columns = np.nonzero(window)
    return columns + left, rows + top


def head_points(mask: np.ndarray, dt: Optional[np.ndarray] = None,
                scale: float = 1.0, min_pixels: Optional[int] = None
                ) -> List[Tuple[int, int, float]]:
    """One point per arrow head, as ``(x, y, thickness)``."""
    if dt is None:
        dt = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    # the head is ~3.4x thicker than a shaft at every scale
    threshold = 3.4 * scale
    if min_pixels is None:
        min_pixels = max(4, int(round(6 * scale * scale)))
    blobs = (dt >= threshold).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(blobs, 8)
    points: List[Tuple[int, int, float]] = []
    for index in range(1, count):
        if stats[index, cv2.CC_STAT_AREA] < min_pixels:
            continue
        columns, rows = component_pixels(labels, stats, index)
        values = dt[rows, columns]
        best = int(np.argmax(values))
        points.append((int(columns[best]), int(rows[best]), float(values[best])))
    return points


def split_component(label_id: int, labels: np.ndarray, stats: np.ndarray,
                    seeds: Sequence[Tuple[int, int, float]]
                    ) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """Partition one ink blob between its heads by multi-source BFS."""
    x0, y0, width, height = (int(v) for v in stats[label_id, :4])
    x1, y1 = x0 + width - 1, y0 + height - 1

    owner = np.full((height, width), -1, np.int32)
    for index, (seed_x, seed_y, _) in enumerate(seeds):
        owner[seed_y - y0, seed_x - x0] = index
    free = (labels[y0:y1 + 1, x0:x1 + 1] == label_id) & (owner < 0)

    for _ in range(max(height, width) + 4):
        if not free.any():
            break
        grew = False
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            shifted = np.full_like(owner, -1)
            y_dst, x_dst = slice(max(0, dy), height + min(0, dy)), slice(max(0, dx), width + min(0, dx))
            y_src, x_src = slice(max(0, -dy), height + min(0, -dy)), slice(max(0, -dx), width + min(0, -dx))
            shifted[y_dst, x_dst] = owner[y_src, x_src]
            take = free & (shifted >= 0)
            if take.any():
                owner[take] = shifted[take]
                free[take] = False
                grew = True
        if not grew:
            break

    parts: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for index, (seed_x, seed_y, _) in enumerate(seeds):
        rows_k, columns_k = np.nonzero(owner == index)
        if len(rows_k):
            parts[index] = (columns_k + x0, rows_k + y0)
        else:
            parts[index] = (np.array([seed_x]), np.array([seed_y]))
    return parts


# --------------------------------------------------------------------------- #
# direction
# --------------------------------------------------------------------------- #
def triangle(scale: float = 1.0) -> np.ndarray:
    """Synthetic head pointing up, with its shaft stub, at ``scale``."""
    radius = max(4, int(round(PATCH_RADIUS * scale)))
    size = 2 * radius + 1
    patch = np.zeros((size, size), np.float32)
    apex_y = radius - HEAD_APEX_OFFSET * scale
    base_y = apex_y + HEAD_HEIGHT * scale
    half_base = HEAD_BASE * scale / 2.0
    points = np.array(
        [(radius - half_base, base_y), (radius + half_base, base_y), (radius, apex_y)],
        np.float32,
    )
    cv2.fillConvexPoly(patch, np.round(points).astype(np.int32), 1.0)

    stub_half = max(1, int(round(SHAFT_HALF_WIDTH * scale)))
    stub_end = min(size - 1, int(round(base_y + SHAFT_STUB * scale)))
    if stub_end > base_y:
        cv2.rectangle(
            patch,
            (radius - stub_half, int(round(base_y))),
            (radius + stub_half, stub_end),
            1.0,
            -1,
        )
    return patch


def rotations(up: np.ndarray) -> Dict[str, np.ndarray]:
    return {
        "U": up,
        "R": cv2.rotate(up, cv2.ROTATE_90_CLOCKWISE),
        "D": cv2.rotate(up, cv2.ROTATE_180),
        "L": cv2.rotate(up, cv2.ROTATE_90_COUNTERCLOCKWISE),
    }


def templates(scale: float = 1.0) -> Dict[str, np.ndarray]:
    return rotations(triangle(scale))


def own_patch(xs: np.ndarray, ys: np.ndarray, x: int, y: int, radius: int) -> np.ndarray:
    """The part of an arrow that falls inside a patch centred on its head."""
    size = 2 * radius + 1
    patch = np.zeros((size, size), np.float32)
    dx, dy = xs - x + radius, ys - y + radius
    inside = (dx >= 0) & (dx <= 2 * radius) & (dy >= 0) & (dy <= 2 * radius)
    patch[dy[inside], dx[inside]] = 1.0
    return patch


def vote(patch: np.ndarray, options: Dict[str, np.ndarray]
         ) -> Tuple[str, Dict[str, float], float]:
    """Best matching direction, all scores, and the margin over the runner-up.

    This is ``TM_CCOEFF_NORMED`` written out by hand: the patch and the template
    are the same size, so one dot product per direction is all it takes.  (The
    OpenCV call has ~1.7 ms of overhead per template, which dominated the whole
    loop on a 140-arrow board.)
    """
    flat = np.asarray(patch, np.float64).ravel()
    flat = flat - flat.mean()
    norm = float(np.sqrt(flat @ flat))
    if norm < 1e-6:
        return "U", {direction: 0.0 for direction in options}, 0.0
    scores: Dict[str, float] = {}
    for direction, template in options.items():
        centred = np.asarray(template, np.float64).ravel()
        centred = centred - centred.mean()
        denominator = norm * float(np.sqrt(centred @ centred))
        scores[direction] = float(centred @ flat) / denominator if denominator else 0.0
    ranked = sorted(scores, key=scores.get, reverse=True)
    return ranked[0], scores, scores[ranked[0]] - scores[ranked[1]]


# --------------------------------------------------------------------------- #
# the board
# --------------------------------------------------------------------------- #
@dataclass(eq=False)
class Arrow:
    """One arrow: where its head is, where it points, and the ink it owns."""

    head: Tuple[int, int]
    direction: str
    confidence: float
    scale: float
    xs: np.ndarray
    ys: np.ndarray
    #: template score for every direction (used to repair a misread deadlock)
    scores: Optional[Dict[str, float]] = None

    @property
    def cx(self) -> int:
        return self.head[0]

    @property
    def cy(self) -> int:
        return self.head[1]

    @property
    def size(self) -> int:
        return len(self.xs)

    def tap_point(self) -> Tuple[int, int]:
        """A point that is unambiguously inside the arrow (its head)."""
        return self.head

    def __str__(self) -> str:
        return f"{self.direction}@({self.head[0]},{self.head[1]}) conf={self.confidence:.2f}"


@dataclass(eq=False)
class Detection:
    """Everything one screenshot said about the board."""

    image: np.ndarray
    mask: np.ndarray
    scale: float
    arrows: List[Arrow]
    bbox: Box

    @property
    def heads(self) -> List[Tuple[int, int]]:
        return [arrow.head for arrow in self.arrows]

    def signature(self) -> Tuple[Box, Tuple[Tuple[int, int], ...]]:
        """A cheap fingerprint used to detect that the camera stopped moving."""
        return self.bbox, tuple(sorted(self.heads))

    def same_view(self, other: "Detection", tolerance: int = 2) -> bool:
        """True when both frames show the same, unmoved board."""
        if other is None or len(self.arrows) != len(other.arrows):
            return False
        if max(abs(a - b) for a, b in zip(self.bbox, other.bbox)) > tolerance:
            return False
        for (x, y), (u, v) in zip(sorted(self.heads), sorted(other.heads)):
            if abs(x - u) > tolerance or abs(y - v) > tolerance:
                return False
        return True

    def describe(self) -> str:
        counts = {d: 0 for d in DIRECTIONS}
        for arrow in self.arrows:
            counts[arrow.direction] = counts.get(arrow.direction, 0) + 1
        spread = " ".join(f"{d}{counts[d]}" for d in DIRECTIONS)
        low = sum(1 for arrow in self.arrows if arrow.confidence < 0.10)
        text = (f"{len(self.arrows)} arrows | scale={self.scale:.2f} | {spread} | "
                f"bbox={self.bbox}")
        return text + (f" | {low} uncertain" if low else "")

    def plan(self):
        """Greedy clear plan (list of waves), or ``None`` on a deadlock."""
        from . import solver

        return solver.plan(self.arrows)


class Detector:
    """Turns a screenshot into :class:`Detection`.

    ``overrides`` lets the agent try one recovery pass with slightly different
    ink/head thresholds without copying the whole configuration.
    """

    def __init__(self, config=None, **overrides):
        self.config = config
        self.overrides = dict(overrides)

    # ---------------------------------------------------------------- helpers
    def _knob(self, name: str, default):
        if name in self.overrides:
            return self.overrides[name]
        return getattr(self.config, name, default) if self.config is not None else default

    def ink(self, image: np.ndarray) -> np.ndarray:
        return ink_mask(
            image,
            roi=self._knob("roi", None),
            exclude_rects=self._knob("exclude_rects", None),
            gray_max=int(self._knob("ink_gray_max", 110)),
            red=bool(self._knob("red_ink", True)),
        )

    def detect(self, image: np.ndarray) -> Detection:
        mask = self.ink(image)
        dt = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        scale = estimate_scale(mask, dt, self._knob("scale_range", (0.5, 4.0)))
        arrows = self.arrows(mask, scale, dt)
        return Detection(image=image, mask=mask, scale=scale,
                         arrows=arrows, bbox=bbox_of(mask))

    def arrows(self, mask: np.ndarray, scale: float,
               dt: Optional[np.ndarray] = None) -> List[Arrow]:
        if dt is None:
            dt = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        options = templates(scale)
        radius = options["U"].shape[0] // 2
        min_pixels = max(30, int(round(float(self._knob("min_arrow_pixels", 60)) * scale * scale)))

        points = head_points(mask, dt, scale=scale,
                             min_pixels=self._knob("min_head_pixels", None))
        _count, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, 8)
        per_blob: Dict[int, List[Tuple[int, int, float]]] = {}
        for point in points:
            blob = int(labels[point[1], point[0]])
            if blob:
                per_blob.setdefault(blob, []).append(point)

        arrows: List[Arrow] = []
        for blob, seeds in per_blob.items():
            if len(seeds) > 1:
                parts = split_component(blob, labels, stats, seeds)
            else:
                parts = {0: component_pixels(labels, stats, blob)}
            for index, (x, y, _thickness) in enumerate(seeds):
                xs, ys = parts[index]
                if len(xs) < min_pixels:
                    continue
                direction, scores, margin = vote(own_patch(xs, ys, x, y, radius), options)
                arrows.append(Arrow(head=(x, y), direction=direction,
                                    confidence=margin, scale=scale, xs=xs, ys=ys,
                                    scores=scores))
        return arrows


# --------------------------------------------------------------------------- #
# debug drawing
# --------------------------------------------------------------------------- #
_PALETTE = {
    "safe": (60, 200, 60),
    "blocked": (150, 150, 150),
    "text": (30, 30, 30),
}


def render_overlay(image: np.ndarray, detection: Detection,
                   safe: Optional[Sequence[Arrow]] = None,
                   title: str = "") -> np.ndarray:
    """Screenshot with every arrow marked: green = safe, grey = blocked."""
    canvas = image.copy()
    safe_ids = {id(arrow) for arrow in safe or ()}
    for arrow in detection.arrows:
        colour = _PALETTE["safe"] if id(arrow) in safe_ids else _PALETTE["blocked"]
        vx, vy = VECTORS[arrow.direction]
        length = int(round(26 * detection.scale))
        x, y = arrow.head
        cv2.arrowedLine(canvas, (x, y), (x + vx * length, y + vy * length),
                        colour, max(1, int(round(detection.scale))), cv2.LINE_AA, 0, 0.35)
        cv2.circle(canvas, (x, y), max(3, int(round(3 * detection.scale))), colour, -1)
    if title:
        cv2.putText(canvas, title, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (255, 255, 255), 5, cv2.LINE_AA)
        cv2.putText(canvas, title, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    _PALETTE["text"], 2, cv2.LINE_AA)
    return canvas
