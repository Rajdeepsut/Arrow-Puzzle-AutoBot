"""Turn the generated Python HUD glyph templates into Java source for the APK.

Run after ``tools/build_counter_templates.py`` changes the font templates::

    python tools/export_android_templates.py
"""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from arrowbot.counter_templates import GLYPH_H, GLYPH_W, TEMPLATES  # noqa: E402

OUT = os.path.join(
    ROOT, "android", "app", "src", "main", "java", "com", "arrowpuzzle",
    "autobot", "core", "DigitTemplates.java",
)


def main() -> int:
    rows = []
    for char in "0123456789":
        art = TEMPLATES[char]
        if len(art) != GLYPH_H or any(len(row) != GLYPH_W for row in art):
            raise SystemExit(f"template {char!r} has the wrong size")
        rows.append("\t\t{ // " + char + "\n" + "\n".join(
            f'\t\t\t"{row}",' for row in art
        ) + "\n\t\t},")
    source = f"""package com.arrowpuzzle.autobot.core;

/** Generated from arrowbot/counter_templates.py - do not edit by hand. */
final class DigitTemplates {{
    static final int WIDTH = {GLYPH_W};
    static final int HEIGHT = {GLYPH_H};

    static final String[][] ART = {{
{os.linesep.join(rows)}
    }};

    private DigitTemplates() {{
    }}
}}
"""
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(source)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
