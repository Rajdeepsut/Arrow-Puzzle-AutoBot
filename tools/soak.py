"""Soak test: play N simulated levels with the production defaults.

    python tools/soak.py --levels 8
    python tools/soak.py --levels 20 --seed 7

Reports per-level seconds, hearts lost and recovery counts.  No phone needed.
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arrowbot.agent import ArrowAgent  # noqa: E402
from arrowbot.config import Config  # noqa: E402
from arrowbot.selftest import _ShiftingSim  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--levels", type=int, default=8)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
    )
    config = Config()
    config.write_debug = False
    config.max_levels = args.levels
    config.start_level = 1

    game = _ShiftingSim(config, cols=11, rows=13, seed=args.seed)
    agent = ArrowAgent(config, game, logger=logging.getLogger("soak"))
    stats = agent.run()

    print()
    print(stats.summary())
    for index, seconds in enumerate(stats.level_times, 1):
        print(f"  level {index:2d}: {seconds:5.2f} s")
    print(f"hearts lost: {game.lives_lost} | sim level: {game.level} | "
          f"lives left: {game.lives}")
    ok = stats.levels_cleared == args.levels and game.lives_lost == 0
    print("SOAK RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
