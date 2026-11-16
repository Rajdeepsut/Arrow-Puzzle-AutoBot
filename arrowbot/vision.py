"""Screen analysis: locate the arrow grid and classify every arrow's direction.

The detector is asset-free.  It does not need to know the level, the grid size
or where the board sits on screen:

1. Build a few candidate binary masks of the screen (dark-on-light,
   light-on-dark, saturation based) inside the configured region of interest.
2. Keep blobs whose size is plausible for a game glyph, then collapse leftovers
   with a median-area filter.
3. Fit a *square lattice* to the blob centres.  This recovers the grid and,
   just as importantly, discards everything that is not part of it: the level
   number, the lives counter, buttons, ad banners.
4. Classify each remaining blob with three independent geometric votes:
   farthest-point, mass skewness and end-width asymmetry.

Everything is expressed in original screenshot pixels, so it is resolution
independent.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import cv2
import numpy as np

from .config import Config
from .solver import DELTAS, DIRECTIONS, format_ascii

Cell = Tuple[int, int]

#: angle (degrees, screen coordinates: x right, y down) of each direction
DIR_ANGLE: Dict[str, float] = {"R": 0.0, "D": 90.0, "L": 180.0, "U": 270.0}


# --------------------------------------------------------------------------- #
# data model
# --------------------------------------------------------------------------- #
@dataclass
class Arrow:
    row: int
    col: int
    direction: str
    confidence: float
    cx: float
    cy: float
    area: int
    bbox: Tuple[int, int, int, int]

    @property
    def cell(self) -> Cell:
        return (self.row, self.col)

    @property
    def known(self) -> bool:
        return self.direction in DELTAS

    def tap_point(self) -> Tuple[int, int]:
        return (int(round(self.cx)), int(round(self.cy)))

    def __str__(self) -> str:
        return f"r{self.row}c{self.col}:{self.direction}({self.confidence:.2f})"


@dataclass
class Detection:
    arrows: List[Arrow] = field(default_factory=list)
    mask_kind: str = "none"
    spacing: float = 0.0
    origin: Tuple[float, float] = (0.0, 0.0)
    dropped: int = 0
    candidates: int = 0

    def __len__(self) -> int:
        return len(self.arrows)

    def __bool__(self) -> bool:
        return bool(self.arrows)

    @property
    def cells(self) -> Dict[Cell, str]:
        return {a.cell: a.direction for a in self.arrows}

    def tappable(self, min_confidence: float) -> List[Arrow]:
        return [
            a for a in self.arrows
            if a.known and a.confidence >= min_confidence
        ]

    def to_ascii(self) -> str:
        return format_ascii(self.cells)

    def describe(self) -> str:
        return (
            f"{len(self.arrows)} arrows | mask={self.mask_kind} "
            f"spacing={self.spacing:.1f}px origin=({self.origin[0]:.1f},"
            f"{self.origin[1]:.1f}) candidates={self.candidates} "
            f"dropped={self.dropped}"
        )


# --------------------------------------------------------------------------- #
# lattice maths (module level: easy to test / reuse)
# --------------------------------------------------------------------------- #
def fit_axis(values: Sequence[float], spacing: float) -> Tuple[float, np.ndarray, np.ndarray]:
    """Fit one axis of the lattice.

    Returns ``(phase, indices, residuals)`` where ``phase + index * spacing``
    approximates every value.  The phase is the circular mean of ``v % spacing``
    so a handful of outliers cannot drag the whole grid around.
    """
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0 or spacing <= 0:
        empty = np.zeros(0, dtype=int)
        return 0.0, empty, np.zeros(0)
    angles = np.mod(v, spacing) / spacing * 2.0 * np.pi
    mean = np.mean(np.exp(1j * angles))
    if abs(mean) < 1e-12:
        phase = 0.0
    else:
        phase = float((np.angle(mean) % (2.0 * np.pi)) / (2.0 * np.pi) * spacing)
    indices, residuals = assign_axis(v, spacing, phase)
    return phase, indices, residuals


def assign_axis(
    values: Sequence[float], spacing: float, phase: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Snap values to a lattice with a *known* phase (no re-estimation).

    Used for the "few arrows left" case, where one or two blobs are far too
    little data to rediscover the grid but plenty to place on the grid we
    already measured.
    """
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0 or spacing <= 0:
        return np.zeros(0, dtype=int), np.zeros(0)
    signed = np.mod(v - phase + spacing / 2.0, spacing) - spacing / 2.0
    return np.rint((v - phase) / spacing).astype(int), np.abs(signed)


def estimate_spacing(
    xs: Sequence[float],
    ys: Sequence[float],
    min_spacing: float = 10.0,
    tol_frac: float = 0.18,
) -> Optional[Tuple[float, float, float]]:
    """Guess the (square) cell size and lattice origin.

    Candidate spacings are derived from the distances between neighbouring blob
    centres, then scored by how many blobs they explain.  Among the candidates
    that explain the same points we prefer the *largest* one so that halves and
    thirds of the real cell size (which always fit too) lose the tie.
    """
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    if xs.size < 4:
        return None

    candidates: Set[float] = set()
    for axis in (xs, ys):
        unique = np.unique(np.round(np.sort(axis), 1))
        if unique.size < 2:
            continue
        for diff in np.diff(unique):
            diff = float(diff)
            if diff < min_spacing:
                continue
            for divisor in (1.0, 2.0, 3.0):
                candidates.add(round(diff / divisor, 2))

    results: List[Tuple[int, float, float, float, float]] = []
    for spacing in sorted(candidates):
        if spacing < min_spacing:
            continue
        ox, _, dx = fit_axis(xs, spacing)
        oy, _, dy = fit_axis(ys, spacing)
        tol = max(3.5, tol_frac * spacing)
        inliers = (dx <= tol) & (dy <= tol)
        count = int(inliers.sum())
        if count < 4:
            continue
        residual = float(np.mean(np.concatenate([dx[inliers], dy[inliers]])))
        results.append((count, float(spacing), residual, float(ox), float(oy)))

    if not results:
        return None

    best_count = max(r[0] for r in results)
    near = [r for r in results if r[0] >= best_count - 1]
    best_residual = min(r[2] for r in near)
    good = [r for r in near if r[2] <= best_residual * 2.0 + 1.0]
    good.sort(key=lambda r: (r[1], -r[0]))
    _, spacing, _, ox, oy = good[-1]
    return spacing, ox, oy


def angle_to_direction(angle: float) -> Tuple[str, float]:
    """Snap an angle in degrees to the nearest cardinal direction."""
    best, best_err = "R", 361.0
    for direction, target in DIR_ANGLE.items():
        err = abs((angle - target + 180.0) % 360.0 - 180.0)
        if err < best_err:
            best, best_err = direction, err
    return best, best_err


def orientation_votes(filled: np.ndarray) -> List[Tuple[np.ndarray, float]]:
    """Three independent guesses about where an arrow points.

    Each vote is ``(unit-ish vector, weight)``.  The three methods fail in
    different ways, so agreement is a meaningful confidence signal.

    a) farthest point from the centroid - the tip of a real arrow glyph
    b) third moment along the principal axis - mass piles up at the *tail*
    c) end width - the tip end of an arrow is narrower than the flat tail end
    """
    rows, cols = np.nonzero(filled)
    if rows.size < 16:
        return []
    points = np.stack([cols, rows], axis=1).astype(np.float64)
    centre = points.mean(axis=0)
    rel = points - centre
    try:
        values, vectors = np.linalg.eigh(np.cov(rel.T))
    except np.linalg.LinAlgError:  # pragma: no cover - degenerate blob
        return []
    axis = vectors[:, int(np.argmax(values))]
    normal = np.array([-axis[1], axis[0]])

    t = rel @ axis
    q = rel @ normal

    votes: List[Tuple[np.ndarray, float]] = []

    # (a) farthest point
    dist = np.linalg.norm(rel, axis=1)
    votes.append((rel[int(np.argmax(dist))], 1.0))

    # (b) skewness: a heavier tail means the tip points the other way
    skew = float(np.mean(t ** 3))
    if abs(skew) > 1e-9:
        votes.append((math.copysign(1.0, -skew) * axis, 0.8))

    # (c) end width: compare the width of the two extreme slices
    lo, hi = float(t.min()), float(t.max())
    span = hi - lo
    if span > 4.0:
        band = max(1.5, 0.06 * span)
        width_hi = float(np.mean(np.abs(q[t >= hi - band])))
        width_lo = float(np.mean(np.abs(q[t <= lo + band])))
        if abs(width_hi - width_lo) > 1e-6:
            votes.append((math.copysign(1.0, width_lo - width_hi) * axis, 1.2))
    return votes


def classify_orientation(filled: np.ndarray) -> Tuple[Optional[str], float]:
    """Turn the vote list into ``(direction, confidence in 0..1)``."""
    votes = orientation_votes(filled)
    if not votes:
        return None, 0.0

    scores = {d: 0.0 for d in DIRECTIONS}
    total = 0.0
    for vector, weight in votes:
        vector = np.asarray(vector, dtype=np.float64).ravel()
        norm = float(math.hypot(vector[0], vector[1]))
        if norm < 1e-9:
            continue
        vector = vector / norm
        angle = math.degrees(math.atan2(vector[1], vector[0])) % 360.0
        direction, err = angle_to_direction(angle)
        total += weight
        if err <= 35.0:
            scores[direction] += weight * (1.0 - err / 90.0)

    if total <= 0.0:
        return None, 0.0
    best = max(DIRECTIONS, key=lambda d: scores[d])
    if scores[best] <= 1e-9:
        return None, 0.0
    return best, float(min(1.0, scores[best] / total))


# --------------------------------------------------------------------------- #
# detector
# --------------------------------------------------------------------------- #
class Detector:
    """Converts a screenshot into a :class:`Detection`.

    The detector remembers the last lattice it fitted.  That matters at the end
    of a level: with only one or two arrows left there is no way to rediscover
    the grid from scratch, but there is no need to either - the board has not
    moved since the last frame.
    """

    def __init__(self, config: Config):
        self.config = config
        self._lattice: Optional[Tuple[float, float, float]] = None  # spacing, ox, oy

    def reset_lattice(self) -> None:
        """Forget the remembered grid (e.g. after a rotation or resize)."""
        self._lattice = None

    # -- public ------------------------------------------------------------
    def detect(self, image: np.ndarray) -> Detection:
        if image is None or image.size == 0:
            return Detection()
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        best: Optional[Detection] = None
        best_key: Optional[Tuple[int, float]] = None

        for kind, mask in self._masks(image):
            blobs, labels = self._blobs(mask)
            if not blobs:
                continue

            # Prefer the grid we measured on an earlier frame: it keeps row and
            # column numbers stable and cannot suddenly pick half a cell size.
            fit = self._fit_remembered(blobs) if self._lattice is not None else None
            if fit is not None and len(fit[0]) < 0.6 * len(blobs):
                fit = None  # the grid moved or the board size changed
            if fit is None and len(blobs) >= 4:
                fit = self._fit(blobs)
            if fit is None:
                continue

            fitted, spacing, origin, dropped = fit
            if not fitted:
                continue

            arrows: List[Arrow] = []
            for blob, (row, col) in fitted:
                direction, confidence = self._classify(labels, blob)
                if direction is None:
                    direction, confidence = "?", 0.0
                arrows.append(
                    Arrow(
                        row=row,
                        col=col,
                        direction=direction,
                        confidence=confidence,
                        cx=float(blob["cx"]),
                        cy=float(blob["cy"]),
                        area=int(blob["area"]),
                        bbox=blob["bbox"],
                    )
                )
            if not arrows:
                continue

            confidences = [a.confidence for a in arrows if a.known]
            mean_conf = sum(confidences) / len(confidences) if confidences else 0.0
            key = (len(arrows), round(mean_conf, 3))
            if best_key is None or key > best_key:
                best_key = key
                best = Detection(
                    arrows=arrows,
                    mask_kind=kind,
                    spacing=spacing,
                    origin=origin,
                    dropped=dropped,
                    candidates=len(blobs),
                )
        if best is not None and best.spacing > 0:
            # keep the grid of the winning mask for the next frame
            self._lattice = (best.spacing, best.origin[0], best.origin[1])
        return best or Detection()

    # -- masks -------------------------------------------------------------
    def _roi_mask(self, height: int, width: int) -> np.ndarray:
        mask = np.zeros((height, width), np.uint8)
        x0, y0, x1, y1 = self._rect_px(self.config.roi, height, width)
        mask[y0:y1, x0:x1] = 255
        for rect in self.config.exclude_rects:
            rx0, ry0, rx1, ry1 = self._rect_px(rect, height, width)
            mask[ry0:ry1, rx0:rx1] = 0
        # never trust a one pixel border: PNG edge artefacts live there
        mask[0:2, :] = 0
        mask[-2:, :] = 0
        mask[:, 0:2] = 0
        mask[:, -2:] = 0
        return mask

    @staticmethod
    def _rect_px(rect: Sequence[float], height: int, width: int) -> Tuple[int, int, int, int]:
        x0, y0, x1, y1 = (float(v) for v in rect)
        return (
            max(0, int(round(x0 * width))),
            max(0, int(round(y0 * height))),
            min(width, int(round(x1 * width))),
            min(height, int(round(y1 * height))),
        )

    def _masks(self, image: np.ndarray) -> Iterable[Tuple[str, np.ndarray]]:
        height, width = image.shape[:2]
        roi = self._roi_mask(height, width)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        _, dark_on_light = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
        )
        _, light_on_dark = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
        )
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        _, saturated = cv2.threshold(
            hsv[:, :, 1], 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
        )

        kernel = None
        if self.config.morph_close >= 3:
            size = int(self.config.morph_close)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))

        for kind, raw in (
            ("dark-on-light", dark_on_light),
            ("light-on-dark", light_on_dark),
            ("saturated", saturated),
        ):
            mask = cv2.bitwise_and(raw, roi)
            if kernel is not None:
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            yield kind, mask

    # -- blobs -------------------------------------------------------------
    def _blobs(self, mask: np.ndarray) -> Tuple[List[dict], np.ndarray]:
        height, width = mask.shape
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask, 8, cv2.CV_32S
        )
        screen_area = float(height * width)
        area_min = max(24.0, self.config.min_blob_area_frac * screen_area)
        area_max = max(area_min * 4.0, self.config.max_blob_area_frac * screen_area)

        blobs: List[dict] = []
        for label in range(1, count):
            x, y, bw, bh, area = (int(v) for v in stats[label])
            if area < area_min or area > area_max:
                continue
            if bw < 5 or bh < 5:
                continue
            aspect = bw / float(bh)
            if aspect < 0.33 or aspect > 3.0:
                continue
            blobs.append(
                {
                    "label": label,
                    "cx": float(centroids[label][0]),
                    "cy": float(centroids[label][1]),
                    "area": area,
                    "bbox": (x, y, bw, bh),
                }
            )
        if len(blobs) < 2:
            return blobs, labels

        # a glyph of a different size (a digit, a button icon) is probably not
        # part of the grid - keep everything within a generous band of the median
        median = float(np.median([b["area"] for b in blobs]))
        blobs = [b for b in blobs if 0.25 * median <= b["area"] <= 4.0 * median]
        return blobs, labels

    # -- lattice -----------------------------------------------------------
    def _fit(
        self, blobs: Sequence[dict]
    ) -> Optional[Tuple[List[Tuple[dict, Cell]], float, Tuple[float, float], int]]:
        """Discover the grid from scratch (needs a decent number of blobs)."""
        xs = np.array([b["cx"] for b in blobs], dtype=np.float64)
        ys = np.array([b["cy"] for b in blobs], dtype=np.float64)
        estimate = estimate_spacing(xs, ys)
        if estimate is None:
            return None
        spacing, ox, oy = estimate
        fitted, dropped = self._assign(blobs, spacing, ox, oy)
        if len(fitted) < 4:
            return None
        return fitted, float(spacing), (float(ox), float(oy)), dropped

    def _fit_remembered(
        self, blobs: Sequence[dict]
    ) -> Optional[Tuple[List[Tuple[dict, Cell]], float, Tuple[float, float], int]]:
        """Place blobs on the grid measured on an earlier frame."""
        if self._lattice is None:
            return None
        spacing, ox, oy = self._lattice
        fitted, dropped = self._assign(blobs, spacing, ox, oy)
        if not fitted:
            return None
        return fitted, float(spacing), (float(ox), float(oy)), dropped

    def _assign(
        self, blobs: Sequence[dict], spacing: float, ox: float, oy: float
    ) -> Tuple[List[Tuple[dict, Cell]], int]:
        tol = max(4.0, self.config.spacing_tolerance * spacing)
        xs = np.array([b["cx"] for b in blobs], dtype=np.float64)
        ys = np.array([b["cy"] for b in blobs], dtype=np.float64)
        ix, dx = assign_axis(xs, spacing, ox)
        iy, dy = assign_axis(ys, spacing, oy)

        kept: Dict[Cell, int] = {}
        dropped = 0
        for index in range(len(blobs)):
            if dx[index] > tol or dy[index] > tol:
                dropped += 1
                continue
            cell = (int(iy[index]), int(ix[index]))
            previous = kept.get(cell)
            if previous is None:
                kept[cell] = index
            elif blobs[index]["area"] > blobs[previous]["area"]:
                kept[cell] = index
                dropped += 1
            else:
                dropped += 1

        fitted = [(blobs[index], cell) for cell, index in kept.items()]
        fitted.sort(key=lambda item: item[1])
        return fitted, dropped

    # -- glyph -------------------------------------------------------------
    def _classify(self, labels: np.ndarray, blob: dict) -> Tuple[Optional[str], float]:
        x, y, bw, bh = blob["bbox"]
        pad = 2
        height, width = labels.shape
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(width, x + bw + pad), min(height, y + bh + pad)

        patch = (labels[y0:y1, x0:x1] == blob["label"]).astype(np.uint8)
        if int(patch.sum()) < 16:
            return None, 0.0
        patch = cv2.morphologyEx(patch * 255, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

        contours, _ = cv2.findContours(patch, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, 0.0
        contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(contour) < 12:
            return None, 0.0

        filled = np.zeros_like(patch)
        cv2.drawContours(filled, [contour], -1, 255, thickness=cv2.FILLED)
        return classify_orientation(filled)


# --------------------------------------------------------------------------- #
# debug rendering
# --------------------------------------------------------------------------- #
def render_overlay(
    image: np.ndarray,
    detection: Detection,
    safe_cells: Optional[Set[Cell]] = None,
    min_confidence: float = 0.6,
    title: Optional[str] = None,
) -> np.ndarray:
    """Annotated copy of the screenshot (written to the debug directory)."""
    out = image.copy()
    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    safe = safe_cells or set()

    for arrow in detection.arrows:
        x, y = int(round(arrow.cx)), int(round(arrow.cy))
        if arrow.cell in safe:
            colour = (0, 170, 0)
        elif not arrow.known or arrow.confidence < min_confidence:
            colour = (0, 165, 255)
        else:
            colour = (0, 0, 220)
        cv2.circle(out, (x, y), 7, colour, 2, cv2.LINE_AA)
        cv2.putText(
            out,
            f"{arrow.direction}{arrow.confidence * 9:.0f}",
            (x + 9, y - 9),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            colour,
            1,
            cv2.LINE_AA,
        )

    header = title or detection.describe()
    cv2.rectangle(out, (0, 0), (out.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(
        out, header, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA
    )

    if detection.spacing > 0 and detection.arrows:
        step = detection.spacing
        origin_x, origin_y = detection.origin
        rmax = max(a.row for a in detection.arrows) + 1
        cmax = max(a.col for a in detection.arrows) + 1
        rmin = min(a.row for a in detection.arrows)
        cmin = min(a.col for a in detection.arrows)
        # draw cell borders (half a cell away from the arrow centres)
        for row in range(rmin, rmax + 1):
            y = int(round(origin_y + (row - 0.5) * step))
            if 0 <= y < out.shape[0]:
                cv2.line(out, (0, y), (out.shape[1], y), (200, 200, 200), 1)
        for col in range(cmin, cmax + 1):
            x = int(round(origin_x + (col - 0.5) * step))
            if 0 <= x < out.shape[1]:
                cv2.line(out, (x, 0), (x, out.shape[0]), (200, 200, 200), 1)
    return out
