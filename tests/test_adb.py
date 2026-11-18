"""Raw screencap decoding and UI-hierarchy helpers (no phone needed)."""

from __future__ import annotations

import struct
from typing import Optional

import cv2
import numpy as np

from arrowbot.adb import Adb, UiNode, _parse_bounds


def _raw(width: int, height: int, pixel_format: int, header_words: int = 4,
         stride: Optional[int] = None) -> bytes:
    stride = stride or width
    if pixel_format == 4:
        payload = np.zeros((height, stride), np.uint16)
    else:
        payload = np.zeros((height, stride, 4 if pixel_format in (1, 2, 5) else 3),
                           np.uint8)
    header = struct.pack("<" + "I" * header_words, width, height, pixel_format,
                         *( [0] * (header_words - 3)))
    return header + payload.tobytes()


def test_raw_rgba_frame_is_decoded_to_bgr():
    width, height = 4, 3
    pixels = np.zeros((height, width, 4), np.uint8)
    pixels[1, 2] = (10, 20, 30, 255)          # R, G, B, A
    raw = struct.pack("<IIII", width, height, 1, 0) + pixels.tobytes()

    frame = Adb._decode_raw(raw)
    assert frame is not None and frame.shape == (height, width, 3)
    assert tuple(int(v) for v in frame[1, 2]) == (30, 20, 10)


def test_raw_12_byte_header_and_stride_padding():
    width, height = 5, 2
    pixels = np.zeros((height, width, 4), np.uint8)
    pixels[0, 0] = (1, 2, 3, 255)
    raw = struct.pack("<III", width, height, 1) + pixels.tobytes()

    frame = Adb._decode_raw(raw)
    assert frame is not None and frame.shape == (height, width, 3)
    assert tuple(int(v) for v in frame[0, 0]) == (3, 2, 1)

    # GPU rows are often padded; the size tells us the real stride
    padded = np.zeros((height, width + 3, 4), np.uint8)
    padded[:, :width] = pixels
    raw_padded = struct.pack("<IIII", width, height, 1, 0) + padded.tobytes()
    frame = Adb._decode_raw(raw_padded)
    assert frame is not None and frame.shape == (height, width, 3)


def test_raw_rgb565_frame_is_decoded():
    width, height = 2, 1
    # 0xF800 is pure red in RGB565, 0x07E0 pure green, 0x001F pure blue
    values = np.array([[0xF800, 0x07E0]], np.uint16)
    raw = struct.pack("<IIII", width, height, 4, 0) + values.tobytes()
    frame = Adb._decode_raw(raw)
    assert frame is not None
    assert tuple(int(v) for v in frame[0, 0]) == (0, 0, 255)
    assert tuple(int(v) for v in frame[0, 1]) == (0, 255, 0)


def test_bad_raw_data_is_rejected_quickly():
    assert Adb._decode_raw(b"") is None
    assert Adb._decode_raw(b"not a screencap") is None
    assert Adb._decode_raw(struct.pack("<IIII", 0, 0, 1, 0)) is None
    assert Adb._decode_raw(struct.pack("<IIII", 100, 100, 99, 0)) is None


def _png_bytes() -> bytes:
    image = np.zeros((6, 8, 3), np.uint8)
    image[:] = (7, 8, 9)
    ok, buffer = cv2.imencode(".png", image)
    assert ok
    return buffer.tobytes()


class _RawAdb(Adb):
    def __init__(self, raw_reply: bytes):
        super().__init__(raw=True)
        self.raw_reply = raw_reply
        self.commands = []

    def _run(self, args, timeout=None):
        self.commands.append(list(args))
        if list(args) == ["exec-out", "screencap"]:
            return self.raw_reply
        if list(args) == ["exec-out", "screencap", "-p"]:
            return _png_bytes()
        return b""


def test_screencap_prefers_raw_and_falls_back_to_png():
    raw = _raw(2, 2, 1)
    adb = _RawAdb(raw)
    frame = adb.screencap()
    assert frame.shape == (2, 2, 3)
    assert adb.commands == [["exec-out", "screencap"]]
    assert adb._raw_failures == 0

    adb = _RawAdb(b"garbage")
    frame = adb.screencap()
    assert frame.shape == (6, 8, 3)
    assert ["exec-out", "screencap", "-p"] in adb.commands
    assert adb._raw_failures == 1


def test_ui_nodes_rank_text_by_priority_and_clickability():
    xml = """<hierarchy>
      <node text="Level 33" content-desc="" clickable="false" enabled="true"
            bounds="[0,0][100,100]" />
      <node text="Watch ad" content-desc="" clickable="true" enabled="true"
            bounds="[10,10][900,200]" />
      <node text="Continue" content-desc="" clickable="true" enabled="true"
            bounds="[200,700][520,860]" />
    </hierarchy>"""
    adb = Adb()
    adb.dump_ui = lambda retries=2: xml       # type: ignore[method-assign]
    nodes = adb.find_text(["continue", "watch"])
    assert [node.text for node in nodes] == ["Continue", "Watch ad"]
    assert nodes[0].center == (360, 780)


def test_parse_bounds():
    assert _parse_bounds("[0,14][1080,2392]") == (0, 14, 1080, 2392)
    assert _parse_bounds("bad") is None
    node = UiNode("OK", "", (10, 20, 30, 40), clickable=True)
    assert node.label == "OK" and node.area == 400
