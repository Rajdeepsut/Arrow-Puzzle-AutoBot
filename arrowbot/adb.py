"""Thin, dependency-light wrapper around the Android Debug Bridge."""

from __future__ import annotations

import re
import struct
import subprocess
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

#: keep console windows from flashing up on Windows
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

#: where ``uiautomator dump`` writes its hierarchy on the phone
REMOTE_UI_DUMP = "/sdcard/_arrowbot_ui.xml"

#: Android PixelFormat -> bytes per pixel
_RAW_BYTES_PER_PIXEL = {1: 4, 2: 4, 3: 3, 4: 2, 5: 4}


@dataclass
class UiNode:
    """One accessibility node read with ``uiautomator dump``."""

    text: str
    description: str
    bounds: Tuple[int, int, int, int]
    clickable: bool = False
    enabled: bool = True

    @property
    def label(self) -> str:
        return (self.text or self.description or "").strip()

    @property
    def center(self) -> Tuple[int, int]:
        x0, y0, x1, y1 = self.bounds
        return (x0 + x1) // 2, (y0 + y1) // 2

    @property
    def area(self) -> int:
        x0, y0, x1, y1 = self.bounds
        return max(0, x1 - x0) * max(0, y1 - y0)


class AdbError(RuntimeError):
    """Raised when the adb binary is missing, times out or fails."""


class Adb:
    """Everything the agent needs: screenshots and taps."""

    def __init__(self, adb_path: str = "adb", serial: Optional[str] = None,
                 timeout: float = 30.0, raw: bool = True):
        self.adb_path = adb_path or "adb"
        self.serial = serial
        self.timeout = float(timeout)
        #: try the raw screencap format first (no PNG encode/decode on the phone)
        self.raw = bool(raw)
        self._raw_failures = 0

    # -- plumbing ----------------------------------------------------------
    def _command(self, args: Sequence[str]) -> List[str]:
        command = [self.adb_path]
        if self.serial:
            command += ["-s", self.serial]
        return command + list(args)

    def _run(self, args: Sequence[str], timeout: Optional[float] = None) -> bytes:
        try:
            proc = subprocess.run(
                self._command(args),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout or self.timeout,
                creationflags=_NO_WINDOW,
            )
        except FileNotFoundError as exc:
            raise AdbError(
                f"'{self.adb_path}' was not found. Install Android platform-tools and "
                "either add adb to PATH or set 'adb_path' in config.json."
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise AdbError(f"adb timed out: {' '.join(args)}") from exc
        if proc.returncode != 0:
            message = proc.stderr.decode("utf-8", "replace").strip()
            raise AdbError(message or f"adb exited with {proc.returncode}: {' '.join(args)}")
        return proc.stdout

    # -- device management -------------------------------------------------
    @staticmethod
    def list_devices(adb_path: str = "adb") -> List[Tuple[str, str]]:
        """Return ``[(serial, state), ...]`` from ``adb devices``."""
        try:
            proc = subprocess.run(
                [adb_path or "adb", "devices"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=20,
                creationflags=_NO_WINDOW,
            )
        except FileNotFoundError as exc:
            raise AdbError(f"'{adb_path}' was not found on PATH.") from exc
        except subprocess.TimeoutExpired as exc:
            raise AdbError("'adb devices' timed out") from exc

        devices: List[Tuple[str, str]] = []
        for line in proc.stdout.decode("utf-8", "replace").splitlines():
            line = line.strip()
            if not line or line.startswith("List of devices"):
                continue
            if line.startswith("*"):  # daemon start-up chatter
                continue
            parts = line.split()
            if len(parts) >= 2:
                devices.append((parts[0], parts[1]))
        return devices

    def reconnect(self) -> None:
        try:
            self._run(["reconnect"], timeout=15)
        except AdbError:
            pass
        time.sleep(0.5)

    # -- screen ------------------------------------------------------------
    def screencap(self, retries: int = 3) -> np.ndarray:
        """Grab one frame as a BGR ``numpy`` array.

        ``adb exec-out screencap`` without ``-p`` skips PNG encoding on the
        phone.  It is several times faster on a full 1080x2392 screen.  Some
        adb/device combinations corrupt raw stdout, so after two failures the
        PNG path is used for the rest of the session.
        """
        last_error: Optional[Exception] = None
        for attempt in range(max(1, retries)):
            try:
                if self.raw and self._raw_failures < 2:
                    try:
                        raw = self._run(["exec-out", "screencap"], timeout=25)
                        frame = self._decode_raw(raw)
                        if frame is not None:
                            return frame
                        self._raw_failures += 1
                    except AdbError as exc:
                        self._raw_failures += 1
                        last_error = exc

                raw = self._run(["exec-out", "screencap", "-p"], timeout=25)
                frame = self._decode(raw)
                if frame is not None:
                    return frame

                # Some adb builds still mangle binary stdout on Windows.
                self._run(["shell", "screencap", "-p", "/sdcard/_arrowbot.png"], timeout=25)
                raw = self._run(["exec-out", "cat", "/sdcard/_arrowbot.png"], timeout=25)
                frame = self._decode(raw)
                if frame is not None:
                    return frame
                raise AdbError("screencap returned data that is not a PNG")
            except AdbError as exc:
                last_error = exc
                if attempt + 1 < retries:
                    time.sleep(0.4)
                    if attempt:
                        self.reconnect()
        raise AdbError(f"screencap failed after {retries} attempts: {last_error}")

    @staticmethod
    def _decode(raw: bytes) -> Optional[np.ndarray]:
        if not raw:
            return None
        frame = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is not None:
            return frame
        # last resort: undo CRLF translation performed by a few Windows adb builds
        if b"\r\n" in raw:
            repaired = raw.replace(b"\r\n", b"\n")
            return cv2.imdecode(np.frombuffer(repaired, dtype=np.uint8), cv2.IMREAD_COLOR)
        return None

    @staticmethod
    def _decode_raw(raw: bytes) -> Optional[np.ndarray]:
        """Decode ``screencap``'s native format (header + RGBA pixels).

        Android writes ``width``, ``height`` and ``PixelFormat`` as native
        ``uint32`` values, then rows of pixels.  Android 9+ adds a fourth
        header word (the dataspace), so both a 12- and a 16-byte header are
        accepted.  GPU strides can pad a row, so the stride is derived from
        the payload size and the image is cropped back to ``width``.
        """
        if len(raw) < 12:
            return None
        width, height, pixel_format = struct.unpack_from("<III", raw, 0)
        if not (0 < width <= 10000 and 0 < height <= 10000):
            return None
        bytes_per_pixel = _RAW_BYTES_PER_PIXEL.get(pixel_format)
        if bytes_per_pixel is None:
            return None
        need = width * height * bytes_per_pixel
        for header in (16, 12):
            payload = raw[header:]
            if len(payload) < need:
                continue
            stride = len(payload) // (height * bytes_per_pixel)
            if stride < width:
                continue
            if pixel_format == 4:              # RGB_565
                values = np.frombuffer(
                    payload[: stride * height * 2], dtype=np.uint16
                ).reshape(height, stride)[:, :width].astype(np.uint32)
                blue = (values & 0x1F) * 255 // 31
                green = ((values >> 5) & 0x3F) * 255 // 63
                red = ((values >> 11) & 0x1F) * 255 // 31
                return np.dstack((blue, green, red)).astype(np.uint8)
            rows = np.frombuffer(
                payload[: stride * height * bytes_per_pixel], dtype=np.uint8
            ).reshape(height, stride, bytes_per_pixel)[:, :width, :3]
            if pixel_format == 1:
                return cv2.cvtColor(np.ascontiguousarray(rows), cv2.COLOR_RGBA2BGR)
            if pixel_format == 5:
                return cv2.cvtColor(np.ascontiguousarray(rows), cv2.COLOR_BGRA2BGR)
            return cv2.cvtColor(np.ascontiguousarray(rows), cv2.COLOR_RGB2BGR)
        return None

    # -- accessibility / recovery -----------------------------------------
    def dump_ui(self, retries: int = 2) -> Optional[str]:
        """Return the current UI hierarchy as XML, or ``None``.

        Only used on a failure path: ``uiautomator`` costs ~1 s, so it never
        runs while a level is being solved.
        """
        for attempt in range(max(1, retries)):
            try:
                self._run(["shell", "uiautomator", "dump", REMOTE_UI_DUMP], timeout=20)
                raw = self._run(["exec-out", "cat", REMOTE_UI_DUMP], timeout=20)
                text = raw.decode("utf-8", "replace")
                start = text.find("<hierarchy")
                if start >= 0:
                    return text[start:]
            except AdbError:
                pass
            if attempt + 1 < retries:
                time.sleep(0.25)
        return None

    def ui_nodes(self, retries: int = 2) -> List[UiNode]:
        """All accessibility nodes on screen (text, description, bounds)."""
        xml = self.dump_ui(retries=retries)
        if not xml:
            return []
        try:
            root = ET.fromstring(xml)
        except ET.ParseError:
            return []
        nodes: List[UiNode] = []
        for element in root.iter("node"):
            attrs = element.attrib
            bounds = _parse_bounds(attrs.get("bounds", ""))
            if bounds is None:
                continue
            nodes.append(UiNode(
                text=attrs.get("text", ""),
                description=attrs.get("content-desc", ""),
                bounds=bounds,
                clickable=attrs.get("clickable", "false") == "true",
                enabled=attrs.get("enabled", "true") == "true",
            ))
        return nodes

    def find_text(self, patterns: Sequence[str]) -> List[UiNode]:
        """Nodes whose text or description contains one of ``patterns``.

        Results are ranked by pattern order, then clickability, then area, so
        the first result is the button a human would press.
        """
        wanted = [pattern.lower() for pattern in patterns if pattern]
        if not wanted:
            return []
        ranked = []
        for node in self.ui_nodes():
            label = node.label.lower()
            if not label or not node.enabled:
                continue
            for priority, pattern in enumerate(wanted):
                if pattern in label:
                    ranked.append((priority, 0 if node.clickable else 1,
                                   node.area, node))
                    break
        ranked.sort(key=lambda item: (item[0], item[1], item[2]))
        return [item[3] for item in ranked]

    # -- power / app lifecycle ---------------------------------------------
    def wake(self) -> None:
        """Wake the screen and dismiss a non-secure keyguard (best effort)."""
        for args in (("input", "keyevent", "KEYCODE_WAKEUP"),
                     ("wm", "dismiss-keyguard")):
            try:
                self._run(["shell", *args], timeout=10)
            except AdbError:
                pass

    def keep_awake(self) -> None:
        """Keep the screen on while charging, so ADB stays authorised."""
        try:
            self._run(["shell", "svc", "power", "stayon", "true"], timeout=10)
        except AdbError:
            pass

    def force_stop(self, package: str) -> None:
        self._run(["shell", "am", "force-stop", package], timeout=20)

    def start_app(self, package: str, activity: Optional[str] = None) -> None:
        if activity:
            self._run(["shell", "am", "start", "-n", f"{package}/{activity}"], timeout=30)
        else:
            self._run(["shell", "monkey", "-p", package, "-c",
                       "android.intent.category.LAUNCHER", "1"], timeout=30)

    # -- input -------------------------------------------------------------
    def tap(self, x: int, y: int) -> None:
        self._run(["shell", "input", "tap", str(int(x)), str(int(y))])

    def tap_many(self, points: Sequence[Tuple[float, float]], gap_ms: int = 0,
                 hold_ms: int = 0) -> int:
        """Fallback batch tap: one adb shell call for all the points.

        ``input`` can only tap one point per process, so this spawns a short
        shell script instead of an adb process per tap.  ``hold_ms`` is ignored
        because plain ``input tap`` has no press duration; the scrcpy backend is
        the fast, multi-touch capable path.
        """
        commands = []
        for index, (x, y) in enumerate(points):
            if index and gap_ms > 0:
                commands.append(f"sleep {gap_ms / 1000.0:.3f}")
            commands.append(f"input tap {int(round(x))} {int(round(y))}")
        if not commands:
            return 0
        self._run(["shell", "; ".join(commands)])
        return len(points)

    def swipe(self, x0: int, y0: int, x1: int, y1: int, duration_ms: int = 200) -> None:
        self._run(
            [
                "shell", "input", "swipe",
                str(int(x0)), str(int(y0)), str(int(x1)), str(int(y1)),
                str(int(duration_ms)),
            ]
        )

    def keyevent(self, code: int) -> None:
        self._run(["shell", "input", "keyevent", str(int(code))])

    def shell(self, *args: str) -> str:
        return self._run(["shell", *args]).decode("utf-8", "replace")

    def screen_size(self) -> Tuple[int, int]:
        """``(width, height)`` as reported by ``wm size``."""
        output = self.shell("wm", "size")
        matches = re.findall(r"(\d+)x(\d+)", output)
        if not matches:
            raise AdbError(f"could not parse screen size from {output!r}")
        width, height = matches[-1]
        return int(width), int(height)


def _parse_bounds(value: str) -> Optional[Tuple[int, int, int, int]]:
    """Parse uiautomator's ``[x0,y0][x1,y1]`` bounds."""
    match = re.match(r"\[(-?\d+),(-?\d+)\]\[(-?\d+),(-?\d+)\]", value.strip())
    if not match:
        return None
    return tuple(int(part) for part in match.groups())  # type: ignore[return-value]
