"""End-to-end self-test: needs no phone, no game install, no assets.

It checks three things:

1. the ray-casting rules on hand written boards,
2. the vision pipeline against synthetic screens it has never seen
   (several grid sizes, both polarities, half-empty boards),
3. the full agent loop against :class:`arrowbot.sim.SimGame` - and it asserts
   that the bot *never lost a life* while clearing several levels.
"""

from __future__ import annotations

import logging
import random
import time
from typing import Dict, List, Tuple

from . import solver
from .agent import ArrowAgent
from .config import Config
from .sim import SimGame
from .vision import Detector

Cell = Tuple[int, int]


class Reporter:
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.failures: List[str] = []
        self.checks = 0

    def check(self, name: str, ok: bool, detail: str = "") -> bool:
        self.checks += 1
        if ok:
            if self.verbose:
                print(f"  PASS  {name}")
        else:
            self.failures.append(name)
            print(f"  FAIL  {name}" + (f"  [{detail}]" if detail else ""))
        return ok

    def summary(self) -> bool:
        print()
        if self.failures:
            print(f"{len(self.failures)} of {self.checks} checks FAILED:")
            for name in self.failures:
                print(f"  - {name}")
            return False
        print(f"all {self.checks} checks passed")
        return True


def _normalise(cells: Dict[Cell, str]) -> Dict[Cell, str]:
    if not cells:
        return {}
    rmin = min(r for r, _ in cells)
    cmin = min(c for _, c in cells)
    return {(r - rmin, c - cmin): d for (r, c), d in cells.items()}


def _quiet_logger() -> logging.Logger:
    logger = logging.getLogger("arrowbot.selftest")
    logger.handlers = [logging.NullHandler()]
    logger.propagate = False
    logger.setLevel(logging.CRITICAL)
    return logger


# --------------------------------------------------------------------------- #
# 1. rules
# --------------------------------------------------------------------------- #
def test_rules(rep: Reporter) -> None:
    board = solver.parse_ascii(
        """
        U . .
        . . .
        . D .
        """
    )
    rep.check("parser reads a board", len(board) == 2, str(board))
    rep.check(
        "open lanes are safe",
        set(solver.safe_cells(board)) == {(0, 0), (2, 1)},
        str(solver.safe_cells(board)),
    )

    blocked = solver.parse_ascii(
        """
        . D .
        . . .
        . U .
        """
    )
    rep.check("facing arrows block each other", solver.safe_cells(blocked) == [])
    rep.check("hopeless board reports a deadlock", solver.plan(blocked) is None)

    chain = solver.parse_ascii("U\nU\nU\nU")
    waves = solver.plan(chain)
    rep.check(
        "chain clears one arrow per wave",
        waves == [[(0, 0)], [(1, 0)], [(2, 0)], [(3, 0)]],
        str(waves),
    )
    rep.check("plan validates legally", bool(waves) and solver.simulate(chain, waves))

    wide = solver.parse_ascii(
        """
        R R R
        R R R
        R R R
        """
    )
    plan = solver.plan(wide)
    rep.check(
        "an all-right board clears one column at a time",
        plan
        == [
            [(0, 2), (1, 2), (2, 2)],
            [(0, 1), (1, 1), (2, 1)],
            [(0, 0), (1, 0), (2, 0)],
        ],
        str(plan),
    )

    mixed = solver.parse_ascii(
        """
        R . D
        . U .
        L . R
        """
    )
    plan = solver.plan(mixed)
    rep.check("mixed board is solvable", plan is not None and solver.simulate(mixed, plan))
    rep.check(
        "mixed board needs several waves",
        plan is not None and sum(len(w) for w in plan) == 5,
        str(plan),
    )

    rng = random.Random(4)
    solvable = solver.random_solvable_board(7, 7, rng)
    rep.check(
        "constructed boards are solvable",
        solver.plan(solvable) is not None,
        f"{len(solvable)} arrows",
    )
    rep.check(
        "constructed boards are reasonably dense",
        len(solvable) >= 0.7 * 49,
        f"{len(solvable)}/49 arrows",
    )


# --------------------------------------------------------------------------- #
# 2. vision
# --------------------------------------------------------------------------- #
def _vision_case(
    rep: Reporter,
    rows: int,
    cols: int,
    seed: int = 1,
    drop: float = 0.0,
    dark: bool = False,
) -> None:
    cfg = Config()
    kwargs = {}
    if dark:
        kwargs = {"background": (32, 32, 38), "foreground": (240, 240, 240)}
    game = SimGame(cfg, rows=rows, cols=cols, seed=seed, **kwargs)

    if drop > 0.0:
        rng = random.Random(seed + 991)
        interior = [
            (r, c) for (r, c) in game.board
            if 0 < r < rows - 1 and 0 < c < cols - 1
        ]
        for cell in rng.sample(interior, int(len(interior) * drop)):
            del game.board[cell]

    image = game.screencap()
    detection = Detector(cfg).detect(image)
    got = _normalise(detection.cells)
    want = _normalise(game.board)

    suffix = ""
    if drop:
        suffix += f", {int(drop * 100)}% cleared"
    if dark:
        suffix += ", dark theme"

    ok = got == want
    detail = ""
    if not ok:
        missing = sorted(set(want) - set(got))[:6]
        extra = sorted(set(got) - set(want))[:6]
        wrong = [(c, want[c], got[c]) for c in sorted(set(want) & set(got)) if want[c] != got[c]][:6]
        detail = f"missing={missing} extra={extra} wrong={wrong} ({detection.describe()})"
    rep.check(f"vision {rows}x{cols}{suffix}", ok, detail)

    if not detection.arrows:
        rep.check(f"tap points {rows}x{cols}{suffix}", False, "no arrows detected")
        return

    hits = [game.cell_at(*a.tap_point()) for a in detection.arrows]
    rep.check(
        f"tap points land in distinct cells {rows}x{cols}{suffix}",
        all(h is not None for h in hits) and len(set(hits)) == len(hits),
        f"{sum(h is None for h in hits)} misses",
    )
    confidences = [a.confidence for a in detection.arrows]
    rep.check(
        f"orientation confidence {rows}x{cols}{suffix}",
        min(confidences) >= 0.6,
        f"min={min(confidences):.2f}",
    )


def test_vision(rep: Reporter) -> None:
    for rows, cols in ((5, 5), (6, 6), (7, 7), (8, 8), (6, 7)):
        _vision_case(rep, rows, cols, seed=rows * 31 + cols)
    _vision_case(rep, 6, 6, seed=5, drop=0.5)
    _vision_case(rep, 8, 8, seed=9, drop=0.35)
    _vision_case(rep, 6, 6, seed=11, dark=True)
    test_end_game(rep)


def test_end_game(rep: Reporter) -> None:
    """The last two arrows of a level must still be found and read correctly."""
    cfg = Config()
    game = SimGame(cfg, rows=6, cols=6, seed=33)
    detector = Detector(cfg)

    full = detector.detect(game.screencap())
    rep.check("end game: full board seen first", len(full) >= 20, full.describe())

    for keep in (2, 1):
        intact = {cell: direction for cell, direction in game.board.items()}
        chosen = sorted(intact)[:keep]
        game.board = {cell: intact[cell] for cell in chosen}
        detection = detector.detect(game.screencap())
        ok = len(detection.arrows) == keep and _normalise(detection.cells) == _normalise(game.board)
        rep.check(
            f"end game: {keep} arrow(s) left are still detected",
            ok,
            f"{detection.describe()} expected={_normalise(game.board)} got={_normalise(detection.cells)}",
        )
        game.board = intact


# --------------------------------------------------------------------------- #
# 3. the real thing, minus the phone
# --------------------------------------------------------------------------- #
def _fast_config(levels: int) -> Config:
    """A config tuned for the simulator: no real sleeps, no debug images."""
    cfg = Config()
    cfg.write_debug = False
    cfg.max_levels = levels
    cfg.chunk_size = 3
    cfg.tap_delay_ms = 0
    cfg.settle_poll_ms = 1
    cfg.settle_timeout_ms = 60
    cfg.advance_wait_ms = 120
    return cfg


def test_agent(rep: Reporter, levels: int = 3) -> None:
    cfg = _fast_config(levels)
    game = SimGame(cfg, rows=6, cols=6, seed=21)
    agent = ArrowAgent(cfg, game, logger=_quiet_logger())
    stats = agent.run()

    rep.check(
        f"agent cleared {levels} simulated levels",
        stats.levels_cleared == levels,
        stats.summary(),
    )
    rep.check(
        "agent never lost a life",
        game.lives_lost == 0,
        f"{game.lives_lost} arrows were tapped while blocked",
    )
    rep.check(
        "agent advanced past the win screen",
        game.level >= levels + 1,
        f"sim level counter is {game.level}",
    )


class _ShiftingSim(SimGame):
    """Emulates the real game: later levels use a different grid size."""

    def _advance(self) -> None:
        if time.monotonic() - self._win_at < 0.2:
            return
        self.level += 1
        self.reshape(self.rng.randint(4, 9), self.rng.randint(4, 9))


def test_grid_changes(rep: Reporter, levels: int = 5) -> None:
    cfg = _fast_config(levels)
    game = _ShiftingSim(cfg, rows=5, cols=5, seed=77)
    agent = ArrowAgent(cfg, game, logger=_quiet_logger())
    stats = agent.run()

    rep.check(
        f"agent survives {levels} grid size changes",
        stats.levels_cleared == levels,
        stats.summary(),
    )
    rep.check(
        "no lives lost while the grid resized",
        game.lives_lost == 0,
        f"{game.lives_lost} blocked taps",
    )


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #
def run(levels: int = 3, verbose: bool = True) -> bool:
    rep = Reporter(verbose=verbose)
    print("rules")
    test_rules(rep)
    print("vision")
    test_vision(rep)
    print("agent (simulated phone)")
    test_agent(rep, levels=levels)
    print("agent with changing grid sizes")
    test_grid_changes(rep)
    return rep.summary()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(0 if run() else 1)
