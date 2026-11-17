"""Tests for the scrcpy wire format and the device wrappers (no phone needed)."""

from __future__ import annotations

import struct

from arrowbot.adb import Adb
from arrowbot.config import Config
from arrowbot.scrcpy import ACTION_DOWN, ACTION_UP, MSG_TOUCH, ScrcpyControl, ScrcpyDevice


def _packets(control: ScrcpyControl):
    packets = []
    control._send = packets.append          # type: ignore[method-assign]
    return packets


def test_a_tap_is_two_well_formed_touch_messages():
    control = ScrcpyControl()
    packets = _packets(control)
    control.tap(120, 240)

    assert len(packets) == 2
    assert all(len(packet) == 32 for packet in packets)
    down = struct.unpack(">BBqiiHHHII", packets[0])
    up = struct.unpack(">BBqiiHHHII", packets[1])
    assert down[0] == MSG_TOUCH and down[1] == ACTION_DOWN
    assert up[0] == MSG_TOUCH and up[1] == ACTION_UP
    assert (down[3], down[4]) == (120, 240)
    assert (up[3], up[4]) == (120, 240)
    assert down[2] == up[2] == 0          # Android expects the first finger to be 0


def test_tap_many_uses_one_down_up_pair_per_point():
    control = ScrcpyControl()
    packets = _packets(control)
    count = control.tap_many([(10, 20), (30, 40), (50, 60)], gap_ms=0)
    assert count == 3
    assert len(packets) == 6
    assert [packet[1] for packet in packets] == [ACTION_DOWN, ACTION_UP] * 3


def test_pinch_uses_two_pointers_and_releases_them():
    control = ScrcpyControl()
    packets = _packets(control)
    control.pinch(500, 700, 280, 70, steps=4, duration_ms=0)

    down = [struct.unpack(">BBqiiHHHII", packet) for packet in packets
            if packet[1] == ACTION_DOWN]
    up = [struct.unpack(">BBqiiHHHII", packet) for packet in packets
          if packet[1] == ACTION_UP]
    assert {event[2] for event in down} == {0, 1}
    assert {event[2] for event in up} == {0, 1}
    # fingers start apart and end together
    xs = sorted(event[3] for event in down)
    end_xs = sorted(event[3] for event in up)
    assert xs[1] - xs[0] > end_xs[1] - end_xs[0]


class _FakeAdb:
    adb_path = "adb"
    serial = "test"
    screen = (1080, 2392)

    def __init__(self):
        self.screen = (1080, 2392)

    def screencap(self):
        return "frame"

    def screen_size(self):
        return self.screen


class _FakeControl:
    def __init__(self):
        self.calls = []

    def tap(self, x, y, **kwargs):
        self.calls.append(("tap", x, y, kwargs))

    def tap_many(self, points, **kwargs):
        self.calls.append(("tap_many", list(points), kwargs))
        return len(list(points))

    def pinch(self, *args, **kwargs):
        self.calls.append(("pinch", args, kwargs))

    def swipe(self, *args, **kwargs):
        self.calls.append(("swipe", args, kwargs))

    def close(self):
        self.calls.append(("close",))


def test_scrcpy_device_forwards_taps_with_screen_size():
    adb = _FakeAdb()
    control = _FakeControl()
    device = ScrcpyDevice(adb, control, 1080, 2392, hold_ms=12)

    assert device.screencap() == "frame"
    device.tap(100, 200)
    device.tap_many([(1, 2), (3, 4)], gap_ms=5)
    device.pinch(500, 600, 280, 70, steps=4, duration_ms=10)
    device.close()

    kinds = [call[0] for call in control.calls]
    assert kinds == ["tap", "tap_many", "pinch", "close"]
    assert control.calls[0][3]["screen_width"] == 1080
    assert control.calls[0][3]["hold_ms"] == 12
    assert control.calls[1][2]["gap_ms"] == 5


class _RecordingAdb(Adb):
    def __init__(self):
        super().__init__()
        self.commands = []

    def _run(self, args, timeout=None):
        self.commands.append(list(args))
        return b""


def test_adb_tap_many_uses_a_single_shell_call():
    adb = _RecordingAdb()
    count = adb.tap_many([(10, 20), (30, 40), (50, 60)], gap_ms=0)
    assert count == 3
    assert len(adb.commands) == 1
    script = adb.commands[0][-1]
    assert script.count("input tap") == 3
    assert "input tap 10 20" in script


def test_fast_defaults_are_sane():
    config = Config().validate()
    assert config.fast_mode is True
    assert config.batch_size >= 2
    assert config.batch_wait_ms < 1000
    assert config.auto_zoom is True
    assert config.zoom_start_gap_px > config.zoom_end_gap_px
