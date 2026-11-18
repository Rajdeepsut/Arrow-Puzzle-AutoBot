"""Live device check: prep the phone, zoom out, tap a real batch, measure drift.

Run this the moment the phone reconnects, before trusting a full `main.py run`:

    python tools/device_test.py             # taps 3 provably safe arrows
    python tools/device_test.py --taps 6    # longer batch (still plan-checked)
    python tools/device_test.py --no-tap    # zoom + read only, tap nothing
    python tools/device_test.py --recovery  # UI dump + app restart smoke test

Every tapped arrow comes from the solver's own clear order, so a heart can only
be lost if the pixel model is wrong or the camera moves mid-batch - which is
exactly what this script measures.
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import sys
import time

import cv2

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arrowbot import solver  # noqa: E402
from arrowbot.adb import Adb  # noqa: E402
from arrowbot.agent import ArrowAgent  # noqa: E402
from arrowbot.config import Config  # noqa: E402
from arrowbot.scrcpy import ScrcpyDevice  # noqa: E402

DEBUG = ROOT / "debug"


def log(message: str) -> None:
    print(message, flush=True)


def prep_phone(adb: Adb, agent: ArrowAgent, config: Config) -> None:
    """Wake the screen, keep it on, launch the game and get to a level."""
    adb.keep_awake()
    adb.wake()
    adb.start_app(config.game_package, config.game_activity or None)
    time.sleep(3.0)

    detection = agent.detect_once(retries=3)
    if agent.looks_like_board(detection):
        log(f"level already open: {detection.describe()}")
        return

    log("no board yet - trying a popup button and the known menu positions")
    if agent.dismiss_overlays():
        time.sleep(1.0)
    detection = agent.detect_once(retries=2)
    if not agent.looks_like_board(detection):
        agent.blind_recovery()
        detection = agent.detect_once(retries=2)
    if agent.looks_like_board(detection):
        log(f"level opened: {detection.describe()}")
    else:
        log("could not find a board - open the game by hand and rerun")


def save(image, name: str) -> None:
    DEBUG.mkdir(exist_ok=True)
    path = DEBUG / name
    cv2.imwrite(str(path), image)
    log(f"  saved {path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--taps", type=int, default=3, help="arrows in the test batch")
    parser.add_argument("--no-tap", action="store_true", help="never tap an arrow")
    parser.add_argument("--no-zoom", action="store_true", help="skip the pinch test")
    parser.add_argument("--recovery", action="store_true",
                        help="smoke-test UI dump, popup dismissal and app restart")
    parser.add_argument("--serial", default=None)
    parser.add_argument("--adb", default=None,
                        help="adb binary (default: config.json adb_path)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s")
    config = Config.load(str(ROOT / "config.json"))
    config.write_debug = False
    adb_path = args.adb or config.adb_path

    devices = [serial for serial, state in Adb.list_devices(adb_path) if state == "device"]
    if not devices:
        log("no device attached - connect USB, unlock the phone, accept the RSA prompt")
        return 2
    config.serial = args.serial or config.serial or devices[0]
    config.validate()
    log(f"device: {config.serial}")

    adb = Adb(adb_path, config.serial, raw=config.raw_screencap)
    with ScrcpyDevice.open(adb, config) as device:
        log(f"scrcpy: {device.control.device_name!r} "
            f"{device.screen_width}x{device.screen_height}")
        agent = ArrowAgent(config, device, logger=logging.getLogger("arrowbot"))

        prep_phone(adb, agent, config)
        detection = agent.read_stable()
        if detection is None or not agent.looks_like_board(detection):
            if detection is not None:
                save(detection.image, "device_prep.png")
            log("no board detected - open a level by hand and rerun")
            return 1

        log(f"board: {detection.describe()}")
        log(f"clipped at the board edge: {agent.board_clipped(detection)}")
        save(detection.image, "device_before.png")

        if config.auto_zoom and not args.no_zoom:
            detection = agent.ensure_zoom(detection)
            if detection is None:
                log("board never stabilised after the pinch")
                return 1
            log(f"after zoom: {detection.describe()}")
            log(f"clipped now: {agent.board_clipped(detection)}")
            save(detection.image, "device_zoomed.png")

        if args.recovery:
            log("recovery smoke test: UI dump")
            nodes = adb.ui_nodes()
            labels = [node.label for node in nodes if node.label][:8]
            matches = [node.label for node in nodes if node.label
                       and any(word in node.label.lower() for word in config.ui_labels)]
            log(f"  {len(nodes)} nodes, labels: {labels}")
            log(f"  recovery-button candidates (not pressed on a live level): {matches}")
            log("recovery smoke test: restart the game")
            if not agent.restart_app():
                log("  restart failed")
                return 1
            time.sleep(1.0)
            detection = agent.read_stable()
            if detection is None or not agent.looks_like_board(detection):
                log("  no board after the restart - check the phone")
                return 1
            log(f"  board came back: {detection.describe()}")
            save(detection.image, "device_restarted.png")

        order = solver.order(detection.arrows)
        if order is None:
            log("deadlock in the detected board - the model misread something")
            save(detection.image, "device_deadlock.png")
            return 1
        log(f"clear plan: {len(order)} arrows, no deadlock")

        if args.no_tap:
            log("--no-tap: stopping before any touch")
            return 0

        batch = order[: max(1, min(args.taps, len(order)))]
        before_heads = [(arrow.cx, arrow.cy) for arrow in batch]
        hearts_ok, red_before = agent.check_hearts(detection.image)
        log(f"hearts before the batch: {'visible' if hearts_ok else 'hidden'}, "
            f"{red_before} red px")
        log(f"tapping {len(batch)} planned-safe arrows over scrcpy "
            f"(hold {config.tap_hold_ms} ms, gap {config.batch_gap_ms} ms)")
        started = time.monotonic()
        agent.tap_batch(batch)
        elapsed = time.monotonic() - started
        log(f"batch sent in {elapsed * 1000:.0f} ms "
            f"({elapsed / len(batch) * 1000:.1f} ms/tap)")

        time.sleep(max(0.4, config.batch_wait_ms / 1000.0 + 0.3))
        after = agent.detect_once(retries=3)
        if after is None:
            log("could not read the board after the batch")
            return 1
        save(after.image, "device_after.png")

        drift = agent.camera_drift(detection, after)
        if drift is None:
            log("too few survivors to measure camera drift")
        else:
            log(f"camera drift after the batch: {drift:.1f} px "
                f"(limit {config.batch_drift_px} px)")

        gone = 0
        survivors = {(arrow.cx, arrow.cy) for arrow in after.arrows}
        radius = 12 * max(0.5, after.scale)
        for x, y in before_heads:
            if not any(abs(x - u) <= radius and abs(y - v) <= radius for u, v in survivors):
                gone += 1
        log(f"tapped arrows gone: {gone}/{len(batch)} | "
            f"hearts lost this run: {agent.stats.hearts_lost}")
        log(f"arrows now: {len(after.arrows)}")

        hearts_ok, red = agent.check_hearts(after.image)
        log(f"hearts check: {'ok' if hearts_ok else 'A HEART WAS LOST'} ({red} red px)")

        if gone != len(batch) or not hearts_ok:
            log("RESULT: batch tap is NOT reliable on this board - run main.py run --slow")
            return 1
        log("RESULT: zoom + batched scrcpy taps work on the device")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
