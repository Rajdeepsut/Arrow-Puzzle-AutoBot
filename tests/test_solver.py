"""Unit tests for the pure pixel-corridor solver (no phone, no screenshots)."""

from __future__ import annotations

import random

from arrowbot import solver
from arrowbot.config import Config
from arrowbot.sim import PITCH, SimArrow, SimGame, arrow_mask
from arrowbot.vision import VECTORS


def make_arrow(head, direction, cells: int = 2) -> SimArrow:
    """A straight pixel arrow whose head is at ``head``."""
    vx, vy = VECTORS[direction]
    points = [(head[0] - vx * PITCH * (cells - k), head[1] - vy * PITCH * (cells - k))
              for k in range(cells + 1)]
    xs, ys, _ = arrow_mask(points, points[-1], direction)
    return SimArrow(points, direction, xs, ys)


def test_a_lone_arrow_is_free():
    arrow = make_arrow((300, 300), "U")
    assert solver.is_safe(arrow, [arrow])
    assert solver.blockers(arrow, [arrow]) == []


def test_an_arrow_in_front_blocks():
    back = make_arrow((300, 300), "U")
    front = make_arrow((300, 300 - 3 * PITCH), "U")
    assert not solver.is_safe(back, [back, front])
    assert solver.is_safe(front, [back, front])


def test_an_arrow_behind_does_not_block():
    up = make_arrow((300, 300), "U")
    behind = make_arrow((300, 300 + 3 * PITCH), "D")
    assert solver.is_safe(up, [up, behind])


def test_an_arrow_beside_the_corridor_does_not_block():
    up = make_arrow((300, 300), "U")
    side = make_arrow((300 + 3 * PITCH, 300), "U")
    assert solver.is_safe(up, [up, side])


def test_perpendicular_arrows_do_not_block_each_other():
    up = make_arrow((300, 300), "U")
    left = make_arrow((500, 500), "L")
    assert solver.is_safe(up, [up, left])
    assert solver.is_safe(left, [up, left])


def test_a_chain_clears_from_the_front():
    chain = [make_arrow((300, 300 + PITCH * k), "U") for k in range(4)]
    safe = solver.safe_arrows(chain)
    assert [id(arrow) for arrow in safe] == [id(chain[0])]
    waves = solver.plan(chain)
    assert waves is not None and len(waves) == 4
    assert sum(len(wave) for wave in waves) == len(chain)


def test_a_ring_of_four_arrows_is_a_deadlock():
    a = make_arrow((100, 100), "R")
    b = make_arrow((200, 100), "D")
    c = make_arrow((200, 200), "L")
    d = make_arrow((100, 200), "U")
    ring = [a, b, c, d]
    assert solver.safe_arrows(ring) == []
    assert solver.plan(ring) is None


def test_the_corridor_scales_with_the_artwork():
    doubled = make_arrow((300, 300), "U")
    doubled.scale = 2.0
    near = make_arrow((300 + 10, 300 - 2 * PITCH), "U")
    assert solver.is_safe(doubled, [doubled, near]) is False


def test_repair_fixes_one_misread_direction():
    """A deadlocked reading becomes clearable by turning one arrow outward."""
    a = make_arrow((100, 100), "R")
    b = make_arrow((200, 100), "D")
    c = make_arrow((200, 200), "L")
    d = make_arrow((100, 200), "U")
    ring = [a, b, c, d]
    assert solver.plan(ring) is None

    fixes = solver.repair(ring)
    assert fixes, "a one-direction repair should exist"
    original = {arrow: arrow.direction for arrow in ring}
    for arrow, direction in fixes.items():
        arrow.direction = direction
    try:
        assert solver.clearable(ring)
    finally:
        for arrow, direction in original.items():
            arrow.direction = direction


def test_repair_leaves_a_clearable_board_alone():
    chain = [make_arrow((300, 300 + PITCH * k), "U") for k in range(4)]
    assert solver.repair(chain) == {}


def test_repair_uses_template_scores_when_present():
    """The true direction has the best score in this synthetic misread."""
    front = make_arrow((300, 300), "U")       # really points up
    back = make_arrow((300, 400), "U")
    chain = [front, back]
    assert solver.clearable(chain)

    class Scored:
        def __init__(self, arrow, direction, scores, confidence=0.04):
            self.head = arrow.head
            self.direction = direction
            self.xs, self.ys = arrow.xs, arrow.ys
            self.scale = arrow.scale
            self.confidence = confidence
            self.scores = scores

    # the detector read the front arrow as down: now both arrows block each other
    front_scored = Scored(front, "D", {"U": 0.62, "D": 0.55, "R": 0.10, "L": 0.05})
    back_scored = Scored(back, "U", {"U": 0.90, "D": 0.05, "R": 0.03, "L": 0.02})
    assert solver.plan([front_scored, back_scored]) is None

    fixes = solver.repair([front_scored, back_scored])
    assert fixes and fixes.get(front_scored) == "U"


def test_order_taps_every_arrow_once_and_stays_safe():
    chain = [make_arrow((300, 300 + PITCH * k), "U") for k in range(6)]
    order = solver.order(chain)
    assert order is not None and len(order) == len(chain)
    alive = list(chain)
    for arrow in order:
        assert solver.is_safe(arrow, alive)
        alive.remove(arrow)
    assert not alive


def test_plans_are_valid_on_generated_boards():
    """Every generated board must be clearable, and each tap safe when made."""
    config = Config()
    for seed in range(8):
        game = SimGame(config, cols=9, rows=11, seed=seed)
        order = solver.order(game.board)
        assert order is not None, f"seed {seed} generated a dead board"
        alive = list(game.board)
        for arrow in order:
            assert solver.is_safe(arrow, alive)
            alive.remove(arrow)
        assert not alive


def test_random_pixel_arrows_never_produce_an_illegal_plan():
    rng = random.Random(11)
    for _ in range(5):
        width, height = 12, 12
        arrows = []
        used = set()
        for _ in range(25):
            row, column = rng.randrange(height), rng.randrange(width)
            step = rng.choice(((1, 0), (-1, 0), (0, 1), (0, -1)))
            if (row, column) in used or not (0 <= row + 2 * step[0] < height
                                             and 0 <= column + 2 * step[1] < width):
                continue
            direction = ("D" if step[0] > 0 else "U") if step[0] else ("R" if step[1] > 0 else "L")
            head = ((column + 2 * step[1]) * PITCH + PITCH / 2,
                    (row + 2 * step[0]) * PITCH + PITCH / 2)
            arrows.append(make_arrow(head, direction))
            used.add((row, column))
        order = solver.order(arrows)
        if order is None:
            continue
        alive = list(arrows)
        for arrow in order:
            assert solver.is_safe(arrow, alive)
            alive.remove(arrow)
