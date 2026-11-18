"""Read the remaining-arrow counter from the game's HUD.

The counter (an arrow icon plus ``74`` in the top-left pill) is the only
*external* check on the board reading: the game knows how many arrows are
left, and if that number disagrees with the arrows :mod:`arrowbot.vision`
found, the plan cannot be fully trusted.  The agent uses this as a soft gate -
a disagreement falls back to one tap per screenshot rather than stopping.

The HUD font is fixed, so ten 20x32 glyph templates are enough.  They are
generated from real captures by ``tools/build_counter_templates.py`` and
matched by pixel agreement; a low score means "unreadable", never "guess".
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

#: normalised glyph size used by the templates
GLYPH_W = 20
GLYPH_H = 32

#: the counter pill, as screen fractions (left, top, right, bottom)
DEFAULT_ROI: Tuple[float, float, float, float] = (0.02, 0.09, 0.20, 0.126)

#: HUD text is dark slate; the pill itself is a very light grey
GRAY_MAX = 150

#: below this pixel agreement a digit is considered unreadable
MIN_SCORE = 0.82


def _crop(image: np.ndarray, roi: Sequence[float]) -> np.ndarray:
    height, width = image.shape[:2]
    x0 = max(0, min(width - 1, int(round(float(roi[0]) * width))))
    y0 = max(0, min(height - 1, int(round(float(roi[1]) * height))))
    x1 = max(x0 + 1, min(width, int(round(float(roi[2]) * width))))
    y1 = max(y0 + 1, min(height, int(round(float(roi[3]) * height))))
    return image[y0:y1, x0:x1]


def counter_glyphs(image: np.ndarray,
                   roi: Optional[Sequence[float]] = DEFAULT_ROI) -> List[np.ndarray]:
    """Split the counter region into normalised glyph bitmaps, left to right.

    ``roi=None`` treats ``image`` as the already cropped counter region.  The
    wide arrow icon is dropped by its aspect ratio; every remaining blob whose
    size is plausible for a digit becomes one template-sized float mask.
    """
    crop = _crop(image, roi) if roi else image
    if crop.size == 0:
        return []
    grey = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    mask = (grey < GRAY_MAX).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    glyphs: List[Tuple[int, np.ndarray]] = []
    for index in range(1, count):
        x, y, w, h, area = stats[index]
        if area < 15 or h < 12 or w < 4:
            continue
        if w / float(h) > 0.85:          # the arrow icon is nearly square
            continue
        patch = (labels[y:y + h, x:x + w] == index).astype(np.float32)
        glyphs.append((int(x), cv2.resize(patch, (GLYPH_W, GLYPH_H),
                                          interpolation=cv2.INTER_AREA)))
    glyphs.sort(key=lambda item: item[0])
    return [glyph for _x, glyph in glyphs]


def match_digit(glyph: np.ndarray, template: np.ndarray) -> float:
    """Fraction of pixels on which a glyph and a template agree."""
    return float(np.mean((glyph > 0.5) == (template > 0.5)))


_TEMPLATE_CACHE: Optional[dict] = None


def _templates() -> dict:
    """Parse the generated ASCII art once (kept readable in the source)."""
    global _TEMPLATE_CACHE
    if _TEMPLATE_CACHE is None:
        from arrowbot.counter_templates import TEMPLATES

        parsed = {}
        for char, art in TEMPLATES.items():
            array = np.array([[1.0 if pixel == "#" else 0.0 for pixel in row]
                              for row in art], dtype=np.float32)
            if array.shape != (GLYPH_H, GLYPH_W):
                raise ValueError(f"template {char!r} is {array.shape}, "
                                 f"expected {(GLYPH_H, GLYPH_W)}")
            parsed[char] = array
        _TEMPLATE_CACHE = parsed
    return _TEMPLATE_CACHE


def read_counter(image: np.ndarray,
                 roi: Optional[Sequence[float]] = DEFAULT_ROI,
                 min_score: float = MIN_SCORE) -> Optional[int]:
    """Return the counter value, or ``None`` when it cannot be read surely."""
    templates = _templates()

    glyphs = counter_glyphs(image, roi)
    if not glyphs or len(glyphs) > 3:
        return None
    text = ""
    for glyph in glyphs:
        char, score = max(((char, match_digit(glyph, template))
                           for char, template in templates.items()),
                          key=lambda item: item[1])
        if score < min_score:
            return None
        text += char
    try:
        return int(text)
    except ValueError:
        return None
