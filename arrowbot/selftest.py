"""End-to-end self-test: needs no phone, no game install, no assets.

Four things are checked:

1. the corridor rules on hand-built pixel arrows,
2. the vision pipeline against synthetic boards it has never seen (several board
   sizes, dark ink, red ink, a zoomed camera, one arrow left),
3. the full agent loop against :class:`arrowbot.sim.SimGame`, asserting that the
   bot *never lost a heart* and did advance to the next level,
4. that a board which never stops moving is never tapped.
"""

from __future__ import annotations

import logging
import struct
import time
from typing import List, Sequence, Tuple

import numpy as np

from . import solver
from .adb import Adb, UiNode
from .agent import ArrowAgent
from .config import Config
from .sim import PITCH, SimArrow, SimGame, arrow_mask
from .vision import VECTORS, Detector


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


def _quiet_logger() -> logging.Logger:
    logger = logging.getLogger("arrowbot.selftest")
    logger.handlers = [logging.NullHandler()]
    logger.propagate = False
    logger.setLevel(logging.CRITICAL)
    return logger


def make_arrow(head: Tuple[int, int], direction: str, cells: int = 2) -> SimArrow:
    """A straight pixel arrow whose head is at ``head``."""
    vx, vy = VECTORS[direction]
    points = [(head[0] - vx * PITCH * (cells - k), head[1] - vy * PITCH * (cells - k))
              for k in range(cells + 1)]
    xs, ys, _ = arrow_mask(points, points[-1], direction)
    return SimArrow(points, direction, xs, ys)


# --------------------------------------------------------------------------- #
# 1. rules
# --------------------------------------------------------------------------- #
def test_rules(rep: Reporter) -> None:
    alone = make_arrow((300, 300), "U")
    rep.check("a lone arrow is free", solver.is_safe(alone, [alone]))
    rep.check("blockers of a lone arrow is empty", solver.blockers(alone, [alone]) == [])

    up = make_arrow((300, 300), "U")
    front = make_arrow((300, 300 - 3 * PITCH), "U")
    rep.check("an arrow in front blocks", not solver.is_safe(up, [up, front]))
    rep.check("the arrow in front is free", solver.is_safe(front, [up, front]))

    behind = make_arrow((300, 300 + 3 * PITCH), "D")
    rep.check("an arrow behind does not block", solver.is_safe(up, [up, behind]))

    side = make_arrow((300 + 3 * PITCH, 300), "U")
    rep.check("an arrow beside the corridor does not block",
              solver.is_safe(up, [up, side]))

    left = make_arrow((500, 500), "L")
    rep.check("perpendicular arrows do not block each other",
              solver.is_safe(up, [up, left]) and solver.is_safe(left, [up, left]))

    chain = [make_arrow((300, 300 + PITCH * k), "U") for k in range(4)]
    safe = solver.safe_arrows(chain)
    rep.check("a chain clears from the front", [id(a) for a in safe] == [id(chain[0])],
              str([str(a) for a in safe]))
    waves = solver.plan(chain)
    rep.check("a chain needs one wave per arrow", waves is not None and len(waves) == 4,
              str(waves))
    rep.check("the plan taps every arrow once",
              waves is not None and sum(len(wave) for wave in waves) == len(chain))

    # four arrows chasing each other's tails: a genuine deadlock
    a = make_arrow((100, 100), "R")
    b = make_arrow((200, 100), "D")
    c = make_arrow((200, 200), "L")
    d = make_arrow((100, 200), "U")
    ring = [a, b, c, d]
    rep.check("a ring of four arrows locks up", solver.safe_arrows(ring) == [],
              str([str(x) for x in solver.safe_arrows(ring)]))
    rep.check("a deadlock is reported as such", solver.plan(ring) is None)

    doubled = make_arrow((300, 300), "U")
    doubled.scale = 2.0
    near = make_arrow((300 + 10, 300 - 2 * PITCH), "U")
    rep.check("the corridor scales with the artwork",
              solver.is_safe(doubled, [doubled, near]) is False)

    # a deadlocked reading is repaired by flipping one misread direction
    fixes = solver.repair(ring)
    rep.check("a deadlock has a one-direction repair", bool(fixes), str(fixes))
    original = {arrow: arrow.direction for arrow in ring}
    for arrow, direction in (fixes or {}).items():
        arrow.direction = direction
    repaired = solver.clearable(ring)
    for arrow, direction in original.items():
        arrow.direction = direction
    rep.check("the repaired board clears", repaired)


# --------------------------------------------------------------------------- #
# 2. vision
# --------------------------------------------------------------------------- #
def _truth(game: SimGame) -> List[Tuple[int, int, str]]:
    return [(int(round(game.to_screen(*arrow.path[-1])[0])),
             int(round(game.to_screen(*arrow.path[-1])[1])),
             arrow.direction) for arrow in game.board]


def _compare(rep: Reporter, name: str, game: SimGame, detector: Detector,
             truth: Sequence[Tuple[int, int, str]]) -> bool:
    detection = detector.detect(game.screencap())
    matched = wrong = missing = 0
    for x, y, direction in truth:
        near = [arrow for arrow in detection.arrows
                if abs(arrow.cx - x) <= 7 and abs(arrow.cy - y) <= 7]
        if not near:
            missing += 1
            continue
        matched += 1
        wrong += near[0].direction != direction
    extra = len(detection.arrows) - matched
    ok = missing == 0 and wrong == 0 and extra == 0
    rep.check(name, ok,
              f"missing={missing} wrong={wrong} extra={extra} ({detection.describe()})")
    if ok and detection.arrows:
        margins = min(arrow.confidence for arrow in detection.arrows)
        rep.check(f"{name}: directions are unambiguous", margins >= 0.06, f"min={margins:.2f}")
    return ok


def test_vision(rep: Reporter) -> None:
    cfg = Config()
    detector = Detector(cfg)
    for cols, rows, seed in ((11, 13, 1), (9, 10, 2), (14, 16, 3), (6, 6, 4)):
        game = SimGame(cfg, cols=cols, rows=rows, seed=seed)
        _compare(rep, f"vision {cols}x{rows}", game, detector, _truth(game))

    game = SimGame(cfg, cols=11, rows=13, seed=5)
    red = SimGame(cfg, cols=11, rows=13, seed=5, foreground=(40, 40, 210))
    _compare(rep, "vision: red arrows", red, detector, _truth(red))
    _compare(rep, "vision: navy arrows", game, detector, _truth(game))

    # the camera zooms in as the board empties - the model must follow
    game = SimGame(cfg, cols=11, rows=13, seed=6)
    full = len(game.board)
    for keep in (max(1, full // 2), 4, 1, 0):
        game.board = game.board[:keep]
        if not game.board:
            break
        _compare(rep, f"vision: zoomed, {keep} arrows left", game, detector, _truth(game))

    # tap points must sit on the arrow they belong to
    game = SimGame(cfg, cols=11, rows=13, seed=7)
    detection = detector.detect(game.screencap())
    hits = 0
    for arrow in detection.arrows:
        x, y = arrow.tap_point()
        hits += int(detection.mask[y, x] > 0)
    rep.check("tap points land on ink", hits == len(detection.arrows),
              f"{hits}/{len(detection.arrows)}")

    # a moving board must be recognised as a different view
    first = detector.detect(game.screencap())
    second = detector.detect(game.screencap())
    rep.check("identical frames count as the same view", first.same_view(second))
    game.board = game.board[1:]
    third = detector.detect(game.screencap())
    rep.check("one arrow fewer is a different view", not first.same_view(third))
    rep.check("a detection knows its own signature", isinstance(first.signature(), tuple))


# --------------------------------------------------------------------------- #
# 3. the real thing, minus the phone
# --------------------------------------------------------------------------- #
def _fast_config(levels: int) -> Config:
    cfg = Config()
    cfg.write_debug = False
    cfg.max_levels = levels
    cfg.fast_mode = True
    cfg.batch_size = 8
    cfg.batch_gap_ms = 0
    cfg.batch_wait_ms = 0
    cfg.tap_hold_ms = 0
    cfg.zoom_settle_ms = 0
    cfg.chunk_size = 1
    cfg.tap_wait_ms = 0
    cfg.settle_poll_ms = 10
    cfg.settle_timeout_ms = 3000
    cfg.advance_wait_ms = 50
    cfg.advance_timeout_ms = 5000
    cfg.advance_attempts = 0
    return cfg


def test_agent(rep: Reporter, levels: int = 3) -> None:
    cfg = _fast_config(levels)
    game = SimGame(cfg, cols=11, rows=13, seed=21)
    agent = ArrowAgent(cfg, game, logger=_quiet_logger())
    stats = agent.run()

    rep.check(f"agent cleared {levels} simulated levels",
              stats.levels_cleared == levels, stats.summary())
    rep.check("agent never lost a heart", game.lives_lost == 0,
              f"{game.lives_lost} wrong taps")
    rep.check("agent advanced past the win screen", game.level >= levels + 1,
              f"sim level counter is {game.level}")
    rep.check("every simulated level cleared in under a minute",
              len(stats.level_times) == levels and max(stats.level_times) < 60.0,
              f"times={['%.1f' % t for t in stats.level_times]}")


def test_zoom_out(rep: Reporter, levels: int = 1) -> None:
    """A camera left zoomed in must be pinched back out before any tap."""
    cfg = _fast_config(levels)
    game = SimGame(cfg, cols=11, rows=13, seed=31)
    full = len(game.board)
    game.camera_zoom = 2.8
    detector = Detector(cfg)
    clipped = detector.detect(game.screencap())
    rep.check("a zoomed-in camera clips the board", len(clipped.arrows) < full,
              f"{len(clipped.arrows)}/{full} arrows visible")

    agent = ArrowAgent(cfg, game, logger=_quiet_logger())
    fitted = agent.ensure_zoom(clipped)
    rep.check("zoom-out brings the whole board back",
              fitted is not None and not agent.board_clipped(fitted)
              and len(fitted.arrows) == full,
              fitted.describe() if fitted else "no detection")
    rep.check("zoom-out reaches the game's fitted camera",
              abs(game.camera_zoom - 1.0) < 0.01, f"camera_zoom={game.camera_zoom:.2f}")


class _ShiftingSim(SimGame):
    """Emulates the real game: later levels use a different board size."""

    def _maybe_advance(self) -> None:
        if self._cleared_at is None:
            self._cleared_at = time.monotonic()
            return
        if time.monotonic() - self._cleared_at < self.advance_delay:
            return
        self.level += 1
        self._cleared_at = None
        self.reshape(self.rng.randint(7, 13), self.rng.randint(8, 14))


def test_board_changes(rep: Reporter, levels: int = 3) -> None:
    cfg = _fast_config(levels)
    game = _ShiftingSim(cfg, cols=9, rows=11, seed=77)
    agent = ArrowAgent(cfg, game, logger=_quiet_logger())
    stats = agent.run()
    rep.check(f"agent survives {levels} board size changes",
              stats.levels_cleared == levels, stats.summary())
    rep.check("no hearts lost while the board resized", game.lives_lost == 0,
              f"{game.lives_lost} wrong taps")


# --------------------------------------------------------------------------- #
# 4. recovery: popups, menus, unstable next boards, raw capture
# --------------------------------------------------------------------------- #
class _OverlaySim(SimGame):
    """A board hidden behind a one-button popup."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.popup = True
        self.dismissed = 0
        self.button = (int(self.width * 0.28), int(self.height * 0.45),
                       int(self.width * 0.72), int(self.height * 0.56))

    def find_text(self, patterns):
        if self.popup and any("continue" in pattern for pattern in patterns):
            return [UiNode("CONTINUE", "", self.button, clickable=True)]
        return []

    def tap(self, x: int, y: int) -> None:
        if self.popup:
            x0, y0, x1, y1 = self.button
            if x0 <= x <= x1 and y0 <= y <= y1:
                self.popup = False
                self.dismissed += 1
            return
        super().tap(x, y)


class _MenuSim(SimGame):
    """Starts on a blank menu; the blind "Continue Game" tap opens the board."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.menu = True
        self.menu_taps = 0

    def screencap(self) -> np.ndarray:
        if not self.menu:
            return super().screencap()
        image = np.empty((self.height, self.width, 3), np.uint8)
        image[:] = self.background
        return image

    def tap(self, x: int, y: int) -> None:
        if self.menu:
            if (abs(x - 0.5 * self.width) < 0.2 * self.width
                    and abs(y - 0.744 * self.height) < 0.06 * self.height):
                self.menu = False
                self.menu_taps += 1
            return
        super().tap(x, y)


def test_recovery(rep: Reporter, levels: int = 1) -> None:
    # a popup button is found through uiautomator and pressed once
    cfg = _fast_config(levels)
    cfg.ui_labels = ["continue", "retry"]
    overlay = _OverlaySim(cfg, cols=9, rows=11, seed=51)
    agent = ArrowAgent(cfg, overlay, logger=_quiet_logger())
    agent._frame_shape = (overlay.height, overlay.width)
    rep.check("a popup button is dismissed",
              agent.dismiss_overlays() is True and overlay.dismissed == 1)
    rep.check("a dismissed popup is not pressed again",
              agent.dismiss_overlays() is False)

    # a blank menu is recovered with a blind "Continue Game" tap, then the level
    # is replayed - never skipped
    cfg = _fast_config(levels)
    cfg.max_level_attempts = 1
    cfg.retry_wait_ms = 0
    menu = _MenuSim(cfg, cols=9, rows=11, seed=21)
    agent = ArrowAgent(cfg, menu, logger=_quiet_logger())
    stats = agent.run()
    rep.check("a menu is dismissed and the level is replayed",
              stats.levels_cleared == levels and menu.menu_taps == 1,
              stats.summary())
    rep.check("recovering a menu loses no heart", menu.lives_lost == 0,
              f"{menu.lives_lost} wrong taps")

    # the screenshot that reveals the next level can catch it mid-animation
    cfg = _fast_config(1)
    cfg.settle_timeout_ms = 250
    game = SimGame(cfg, cols=9, rows=10, seed=41)
    detector = Detector(cfg)
    candidate = detector.detect(game.screencap())
    settled_agent = ArrowAgent(cfg, game, logger=_quiet_logger())
    rep.check("a settled next board is accepted",
              settled_agent._stable_next_board(candidate) is not None)

    moving_agent = ArrowAgent(cfg, _JitteringDevice(game.screencap()),
                              logger=_quiet_logger())
    rep.check("an animating next board is not used",
              moving_agent._stable_next_board(candidate) is None)

    # raw screencap pixels decode to BGR without any PNG step
    raw = struct.pack("<IIII", 2, 2, 1, 0) + bytes([255, 0, 0, 255]) * 4
    frame = Adb._decode_raw(raw)
    rep.check("raw screencap decodes to BGR",
              frame is not None and frame.shape == (2, 2, 3)
              and tuple(int(v) for v in frame[0, 0]) == (0, 0, 255),
              "decoder returned %s" % (None if frame is None else str(frame.shape)))


# --------------------------------------------------------------------------- #
# 4. the scrcpy wire format (no phone needed)
# --------------------------------------------------------------------------- #
def test_scrcpy_packets(rep: Reporter) -> None:
    import struct

    from .scrcpy import ACTION_DOWN, ACTION_UP, MSG_TOUCH, ScrcpyControl

    control = ScrcpyControl()
    packets: List[bytes] = []
    control._send = packets.append          # type: ignore[method-assign]
    control.tap(120, 240)
    rep.check("a tap is exactly two 32-byte touch messages",
              len(packets) == 2 and all(len(packet) == 32 for packet in packets),
              f"sizes={[len(packet) for packet in packets]}")

    def fields(packet: bytes):
        return struct.unpack(">BBqiiHHHII", packet)

    down, up = (fields(packet) for packet in packets)
    rep.check("scrcpy down packet is a touch-down at the tapped point",
              down[0] == MSG_TOUCH and down[1] == ACTION_DOWN and down[2] == 0
              and (down[3], down[4]) == (120, 240), str(down))
    rep.check("scrcpy up packet releases the same pointer",
              up[0] == MSG_TOUCH and up[1] == ACTION_UP and up[2] == 0
              and (up[3], up[4]) == (120, 240), str(up))

    packets.clear()
    count = control.tap_many([(10, 20), (30, 40), (50, 60)], gap_ms=0)
    rep.check("tap_many sends one down/up pair per arrow",
              count == 3 and len(packets) == 6
              and [packet[1] for packet in packets] == [ACTION_DOWN, ACTION_UP] * 3,
              f"{len(packets)} packets")


# --------------------------------------------------------------------------- #
# 5. a board that never settles must never be tapped
# --------------------------------------------------------------------------- #
class _JitteringDevice:
    def __init__(self, image: np.ndarray):
        self.image = image
        self.count = 0
        self.taps: List[Tuple[int, int]] = []

    def screencap(self) -> np.ndarray:
        self.count += 1
        return np.roll(self.image, (self.count * 7) % 60, axis=1)

    def tap(self, x: int, y: int) -> None:
        self.taps.append((x, y))


def test_unstable_view(rep: Reporter) -> None:
    cfg = _fast_config(1)
    cfg.settle_timeout_ms = 400
    game = SimGame(cfg, cols=9, rows=10, seed=9)
    device = _JitteringDevice(game.screencap())
    agent = ArrowAgent(cfg, device, logger=_quiet_logger())

    rep.check("a moving view is refused", agent.read_stable() is None)
    rep.check("a moving view is never tapped", agent.play_level() == "unstable")
    rep.check("no tap was sent while moving", device.taps == [], str(device.taps[:4]))


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
    print("zoom-out at level start")
    test_zoom_out(rep)
    print("agent with changing board sizes")
    test_board_changes(rep)
    print("recovery (popups, menus, raw capture)")
    test_recovery(rep)
    print("scrcpy control protocol")
    test_scrcpy_packets(rep)
    print("a board that keeps moving")
    test_unstable_view(rep)
    return rep.summary()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(0 if run() else 1)
