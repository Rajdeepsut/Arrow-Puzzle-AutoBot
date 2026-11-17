#!/usr/bin/env python3
"""Command line interface for the Arrow Puzzle auto-solver.

    python main.py selftest            # verify the install, no phone needed
    python main.py devices             # list connected Android devices
    python main.py shot                # capture once, detect, save an overlay
    python main.py shot --image foo.png
    python main.py run                 # play levels (fast mode, scrcpy input)
    python main.py run --dry-run       # detect + zoom + plan, never tap
    python main.py run --slow          # one tap per screenshot
    python main.py run --backend adb   # no scrcpy (no pinch, slower taps)

Fast mode plans the whole board once and taps it in batches over the scrcpy
control socket: one level typically takes a few seconds after the page loads.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import List, Optional

import cv2

from arrowbot import __version__, selftest, solver
from arrowbot.adb import Adb, AdbError
from arrowbot.agent import ArrowAgent
from arrowbot.config import Config
from arrowbot.scrcpy import ScrcpyDevice, ScrcpyError
from arrowbot.vision import Detector, render_overlay

LOG_FORMAT = "%(asctime)s %(levelname)-7s %(message)s"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def setup_logging(verbose: bool, log_file: Optional[str] = None) -> logging.Logger:
    level = logging.DEBUG if verbose else logging.INFO
    handlers: List[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        directory = os.path.dirname(os.path.abspath(log_file))
        if directory:
            os.makedirs(directory, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(level=level, format=LOG_FORMAT, handlers=handlers, force=True)
    logger = logging.getLogger("arrowbot")
    logger.setLevel(level)
    return logger


def load_config(args: argparse.Namespace) -> Config:
    config = Config.load(getattr(args, "config", None))
    for attribute in ("serial", "max_levels", "start_level", "batch_size"):
        value = getattr(args, attribute, None)
        if value is not None:
            setattr(config, attribute, value)
    fast = getattr(args, "fast_mode", None)
    if fast is not None:
        config.fast_mode = bool(fast)
    if getattr(args, "backend", None):
        config.input_backend = args.backend
    if getattr(args, "no_zoom", False):
        config.auto_zoom = False
    return config.validate()


def resolve_device(config: Config, logger: logging.Logger) -> None:
    devices = Adb.list_devices(config.adb_path)
    if not devices:
        raise SystemExit(
            "No Android device found.\n"
            "  * enable Developer options -> USB debugging\n"
            "  * connect the phone and accept the RSA prompt\n"
            "  * check with:  python main.py devices"
        )
    serials = [serial for serial, state in devices if state == "device"]
    if not serials:
        raise SystemExit(f"Devices are present but not authorised: {devices}")
    if config.serial is None:
        if len(serials) > 1:
            raise SystemExit(
                "Several devices are connected - choose one with --serial:\n  "
                + "\n  ".join(serials)
            )
        config.serial = serials[0]
    elif config.serial not in serials:
        raise SystemExit(f"Device {config.serial} not found. Available: {serials}")
    logger.info("using device %s", config.serial)


def open_device(config: Config, logger: logging.Logger):
    """Return ``(device, closer)``; prefers scrcpy for fast multi-touch input."""
    adb = Adb(config.adb_path, config.serial)
    if config.input_backend != "scrcpy":
        logger.info("input: adb shell (slow, single finger)")
        return adb, (lambda: None)
    try:
        device = ScrcpyDevice.open(adb, config)
    except (ScrcpyError, AdbError, OSError) as exc:
        logger.warning("scrcpy is unavailable (%s) - falling back to adb taps", exc)
        return adb, (lambda: None)
    logger.info("input: scrcpy on %s (%dx%d, real multi-touch)",
                device.control.device_name or "device",
                device.screen_width, device.screen_height)
    return device, device.close


def save_overlay(image, detection, path: str) -> str:
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    safe = solver.safe_arrows(detection.arrows)
    overlay = render_overlay(image, detection, safe=safe,
                             title=f"{detection.describe()}")
    cv2.imwrite(path, overlay)
    return path


def print_detection(detection, verbose: bool = True) -> None:
    print(detection.describe())
    if not detection.arrows:
        print(
            "\nNo arrows found. Things to try:\n"
            "  * open a level so the board is visible\n"
            "  * widen 'roi' in config.json\n"
            "  * zoom the game board out (two fingers)\n"
            "  * run with -v and look at the saved overlay image"
        )
        return
    safe = solver.safe_arrows(detection.arrows)
    print(f"safe right now: {len(safe)}")
    if not verbose:
        return
    for arrow in sorted(detection.arrows, key=lambda a: (a.cy, a.cx)):
        x, y = arrow.tap_point()
        print(f"  {arrow.direction} -> tap=({x},{y}) conf={arrow.confidence:.2f}")
    print()
    waves = solver.plan(detection.arrows)
    if waves is None:
        print("plan: DEAD-LOCKED (or an arrow was misread) - see the overlay")
    else:
        print(f"plan: {len(waves)} wave(s), {sum(len(w) for w in waves)} tap(s)")


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #
def cmd_devices(args, config, logger) -> int:
    try:
        devices = Adb.list_devices(config.adb_path)
    except AdbError as exc:
        print(f"error: {exc}")
        return 2
    if not devices:
        print("no devices attached")
        return 1
    for serial, state in devices:
        print(f"{serial:<24} {state}")
    return 0


def cmd_shot(args, config, logger) -> int:
    if args.image:
        image = cv2.imread(args.image)
        if image is None:
            print(f"error: could not read {args.image}")
            return 2
        source = args.image
    else:
        resolve_device(config, logger)
        image = Adb(config.adb_path, config.serial).screencap()
        source = f"device {config.serial}"

    detector = Detector(config)
    detection = detector.detect(image)
    height, width = image.shape[:2]
    print(f"source: {source} ({width}x{height})")
    print_detection(detection, verbose=not args.quiet)

    path = save_overlay(image, detection, args.out)
    print(f"\noverlay written to {path}")
    return 0 if detection.arrows else 1


def cmd_run(args, config, logger) -> int:
    resolve_device(config, logger)
    device, close = open_device(config, logger)
    agent = ArrowAgent(config, device, logger=logger)

    if config.write_debug:
        os.makedirs(config.debug_dir, exist_ok=True)
        logger.info("debug images go to %s", os.path.abspath(config.debug_dir))

    try:
        stats = agent.run(dry_run=args.dry_run)
    except AdbError as exc:
        logger.error("adb failure: %s", exc)
        return 2
    except KeyboardInterrupt:
        logger.warning("stopped by user")
        return 130
    finally:
        close()

    logger.info("finished: %s", stats.summary())
    return 0


def cmd_selftest(args, config, logger) -> int:
    return 0 if selftest.run(levels=args.levels) else 1


# --------------------------------------------------------------------------- #
# parser
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="arrowbot",
        description="Autonomous solver for 'Arrow Puzzle' (Easybrain).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--config", help="path to config.json (default: ./config.json)")
    parser.add_argument("--log-file", help="also write the log to this file")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    parser.add_argument("--version", action="version", version=f"arrowbot {__version__}")

    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="play levels")
    run.add_argument("--max-levels", type=int, help="stop after this many cleared levels")
    run.add_argument("--start-level", type=int, help="label for the first level (logging only)")
    run.add_argument("--serial", help="adb device serial")
    run.add_argument("--batch-size", type=int, help="arrows tapped per screenshot")
    run.add_argument("--backend", choices=("scrcpy", "adb"),
                     help="input backend (default: scrcpy, adb fallback)")
    run.add_argument("--no-zoom", action="store_true",
                     help="skip the zoom-out pinch at level start")
    mode = run.add_mutually_exclusive_group()
    mode.add_argument("--fast", dest="fast_mode", action="store_true", default=None,
                      help="plan once, tap batches (default)")
    mode.add_argument("--slow", dest="fast_mode", action="store_false",
                      help="one tap per screenshot (for a moving camera)")
    run.add_argument("--dry-run", action="store_true",
                     help="detect, zoom and plan, but never tap")
    run.set_defaults(func=cmd_run)

    shot = sub.add_parser("shot", help="capture one frame, detect, save an overlay")
    shot.add_argument("--serial", help="adb device serial")
    shot.add_argument("--image", help="analyse a saved PNG instead of the phone")
    shot.add_argument("--out", default=os.path.join("debug", "shot.png"))
    shot.add_argument("-q", "--quiet", action="store_true", help="summary only")
    shot.set_defaults(func=cmd_shot)

    devices = sub.add_parser("devices", help="list adb devices")
    devices.set_defaults(func=cmd_devices)

    test = sub.add_parser("selftest", help="run the built-in end-to-end test")
    test.add_argument("--levels", type=int, default=3, help="simulated levels to play")
    test.set_defaults(func=cmd_selftest)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logger = setup_logging(args.verbose, args.log_file)

    if args.command == "selftest":
        return cmd_selftest(args, None, logger)

    config = load_config(args)
    try:
        return args.func(args, config, logger)
    except SystemExit as exc:  # friendly message from resolve_device
        print(exc)
        return 2
    except AdbError as exc:  # adb missing, unauthorised, unplugged, ...
        print(f"error: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
