"""Thin, dependency-light wrapper around the Android Debug Bridge."""

from __future__ import annotations

import subprocess
import time
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

#: keep console windows from flashing up on Windows
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


class AdbError(RuntimeError):
    """Raised when the adb binary is missing, times out or fails."""


class Adb:
    """Everything the agent needs: screenshots and taps."""

    def __init__(self, adb_path: str = "adb", serial: Optional[str] = None, timeout: float = 30.0):
        self.adb_path = adb_path or "adb"
        self.serial = serial
        self.timeout = float(timeout)

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
        """Grab one frame as a BGR ``numpy`` array."""
        last_error: Optional[Exception] = None
        for attempt in range(max(1, retries)):
            try:
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
        import re

        output = self.shell("wm", "size")
        matches = re.findall(r"(\d+)x(\d+)", output)
        if not matches:
            raise AdbError(f"could not parse screen size from {output!r}")
        width, height = matches[-1]
        return int(width), int(height)
