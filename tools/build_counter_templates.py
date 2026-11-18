"""Build the HUD counter templates from saved real captures.

The saved tap sequence ``debug/real/live_t0..t55_s1.png`` was taken while a
56-arrow level was cleared one tap at a time, so frame ``tN`` shows exactly
``56 - N`` in the game's remaining-arrow counter.  ``done1.png`` is a Level 33
board with 74 arrows.  Together they label every digit 0-9.

The output, ``arrowbot/counter_templates.py``, is committed; this script only
needs to run when the HUD font or layout changes.  It validates the generated
templates two ways before writing anything:

* leave-one-out over all 57 frames,
* the two independently captured fixtures ``samples/real_level*.png``.

    python tools/build_counter_templates.py
"""

from __future__ import annotations

import glob
import pathlib
import re
import sys

import cv2
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arrowbot.counter import counter_glyphs, match_digit  # noqa: E402

REAL = ROOT / "debug" / "real"
SAMPLES = ROOT / "samples"
OUTPUT = ROOT / "arrowbot" / "counter_templates.py"


def frame_number(path: str) -> int:
    match = re.search(r"live_t(\d+)_s1\.png$", path)
    return int(match.group(1)) if match else -1


def labelled_frames():
    files = sorted((p for p in glob.glob(str(REAL / "live_t*_s1.png"))
                    if frame_number(p) >= 0), key=frame_number)
    if not files:
        print("no saved tap sequence in debug/real - cannot rebuild templates")
        return []
    samples = [(str(56 - frame_number(path)), cv2.imread(path)) for path in files]
    done = REAL / "done1.png"
    if done.exists():
        samples.append(("74", cv2.imread(str(done))))
    return samples


def build(samples):
    collected = {}
    for text, image in samples:
        glyphs = counter_glyphs(image)
        if len(glyphs) != len(text):
            print(f"  skip count {text}: {len(glyphs)} glyphs")
            continue
        for char, glyph in zip(text, glyphs):
            collected.setdefault(char, []).append(glyph)
    templates = {}
    for char in sorted(collected):
        mean = np.mean(collected[char], axis=0)
        templates[char] = (mean > 0.5).astype(np.float32)
    return templates, collected


def read_with(glyphs, templates, min_score=0.82):
    if not glyphs or len(glyphs) > 3:
        return None, 0.0
    text = ""
    worst = 1.0
    for glyph in glyphs:
        char, score = max(((c, match_digit(glyph, t)) for c, t in templates.items()),
                          key=lambda item: item[1])
        text += char
        worst = min(worst, score)
    if worst < min_score:
        return None, worst
    return int(text), worst


def validate(templates) -> bool:
    ok = True
    samples = labelled_frames()
    print(f"leave-one-out over {len(samples)} frames")
    for index, (text, image) in enumerate(samples):
        others = samples[:index] + samples[index + 1:]
        without, _ = build(others)
        if any(char not in without for char in text):
            continue
        got, score = read_with(counter_glyphs(image), without)
        if str(got) != text:
            ok = False
            print(f"  expected {text} got {got} score {score:.3f}")
    print("  all labels read correctly" if ok else "  MISMATCH")
    for name, expected in (("real_level.png", 56), ("real_level_zoomed.png", 78)):
        image = cv2.imread(str(SAMPLES / name))
        if image is None:
            continue
        got, score = read_with(counter_glyphs(image), templates)
        status = "ok" if got == expected else "WRONG"
        print(f"  {name}: {got} (expected {expected}, score {score:.3f}) {status}")
        ok = ok and got == expected
    return ok


def emit(templates) -> str:
    lines = [
        '"""Digit templates for the remaining-arrow counter (generated).',
        "",
        "Regenerate with ``python tools/build_counter_templates.py`` when the HUD",
        "font changes.  Every template is a 20x32 binary glyph; ``#`` is ink.",
        '"""',
        "",
        "from __future__ import annotations",
        "",
        "GLYPH_W = 20",
        "GLYPH_H = 32",
        "",
        "TEMPLATES = {",
    ]
    for char in sorted(templates):
        lines.append(f'    "{char}": (')
        for row in templates[char]:
            art = "".join("#" if value > 0.5 else "." for value in row)
            lines.append(f'        "{art}",')
        lines.append("    ),")
    lines.append("}")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    samples = labelled_frames()
    if not samples:
        return 1
    print(f"frames: {len(samples)}")
    templates, collected = build(samples)
    print("samples per digit:",
          {char: len(items) for char, items in sorted(collected.items())})
    missing = sorted(set("0123456789") - set(templates))
    if missing:
        print("missing digits:", missing)
        return 1
    if not validate(templates):
        print("validation failed - templates not written")
        return 1
    OUTPUT.write_text(emit(templates), encoding="utf-8", newline="\n")
    print(f"wrote {OUTPUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
