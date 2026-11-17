"""Multi-touch input through the scrcpy server (no root, no app install).

Android's ``input`` command can only inject one finger, and ``sendevent`` is
forbidden to the shell user on modern phones, so a real pinch is impossible with
those.  The scrcpy server, however, runs as the shell user with the
``INJECT_EVENTS`` permission and accepts motion events with **several pointers**
over a local socket.  That is exactly what this module speaks.

The server is started with ``video=false audio=false``, so it opens a single
socket: one dummy byte, the 64-byte device name, then the raw control channel.
A touch message is 32 bytes::

    type(1)=2  action(1)  pointer_id(8)  x(4) y(4) w(2) h(2)
    pressure(2)  action_button(4)  buttons(4)

MotionEvent actions: 0 down, 1 up, 2 move, 5 pointer-down, 6 pointer-up.  The
server turns a second DOWN into POINTER_DOWN by itself, so a pinch is simply
DOWN id=1, DOWN id=2, MOVE id=1/id=2 ..., UP id=1, UP id=2.

Everything here is optional: if scrcpy cannot start, the agent falls back to
``adb shell input tap``.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import socket
import struct
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

SCRCPY_VERSION = "5.0.1"
SCRCPY_SERVER_NAME = f"scrcpy-server-v{SCRCPY_VERSION}"
SCRCPY_SERVER_SHA256 = "764eb6f79811d5211fe9df341120882ba9994c7a61b897d7bf3fb662e53bc536"
SCRCPY_SERVER_URL = (
    f"https://github.com/Genymobile/scrcpy/releases/download/v{SCRCPY_VERSION}/"
    f"{SCRCPY_SERVER_NAME}"
)
DEVICE_JAR = "/data/local/tmp/arrowbot-scrcpy-server.jar"

#: MotionEvent actions
ACTION_DOWN = 0
ACTION_UP = 1
ACTION_MOVE = 2
ACTION_POINTER_DOWN = 5
ACTION_POINTER_UP = 6

#: message type
MSG_TOUCH = 2


class ScrcpyError(RuntimeError):
    """Raised when the scrcpy server or its control socket is unavailable."""


def default_jar_path() -> Path:
    """Where the server jar is cached on this machine."""
    if sys.platform.startswith("win"):
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return base / "arrowbot" / "cache" / SCRCPY_SERVER_NAME


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_server_jar(path: Optional[str] = None, download: bool = True) -> Path:
    """Return a verified server jar, downloading it once if necessary."""
    candidates = []
    if path:
        candidates.append(Path(path))
    candidates.append(Path(__file__).resolve().parent.parent / "vendor" / SCRCPY_SERVER_NAME)
    candidates.append(default_jar_path())
    for candidate in candidates:
        if candidate.is_file() and _sha256(candidate) == SCRCPY_SERVER_SHA256:
            return candidate

    target = Path(path) if path else default_jar_path()
    if not download:
        raise ScrcpyError(
            f"scrcpy server jar not found (looked in {', '.join(map(str, candidates))}); "
            f"download {SCRCPY_SERVER_URL} to {target}"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".part")
    try:
        with urllib.request.urlopen(SCRCPY_SERVER_URL, timeout=60) as response:
            with open(tmp, "wb") as handle:
                shutil.copyfileobj(response, handle)
        if _sha256(tmp) != SCRCPY_SERVER_SHA256:
            raise ScrcpyError("downloaded scrcpy server failed its SHA-256 check")
        tmp.replace(target)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    return target


class ScrcpyControl:
    """A minimal scrcpy control client with real multi-touch support."""

    def __init__(
        self,
        adb_path: str = "adb",
        serial: Optional[str] = None,
        jar: Optional[str] = None,
        port: int = 27183,
        timeout: float = 20.0,
        log_level: str = "error",
        keep_awake: bool = True,
    ):
        self.adb_path = adb_path or "adb"
        self.serial = serial
        self.jar = jar
        self.port = int(port)
        self.timeout = float(timeout)
        self.log_level = log_level
        self.keep_awake = keep_awake
        self.sock: Optional[socket.socket] = None
        self.server: Optional[subprocess.Popen] = None
        self.device_name = ""

    # ------------------------------------------------------------------ adb
    def _adb_command(self, args: Sequence[str]) -> list:
        command = [self.adb_path]
        if self.serial:
            command += ["-s", self.serial]
        return command + list(args)

    def _adb(self, args: Sequence[str], timeout: Optional[float] = None,
             check: bool = True) -> subprocess.CompletedProcess:
        proc = subprocess.run(
            self._adb_command(args),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout or self.timeout,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if check and proc.returncode != 0:
            message = proc.stderr.decode("utf-8", "replace").strip()
            raise ScrcpyError(message or f"adb failed: {' '.join(args)}")
        return proc

    # --------------------------------------------------------------- startup
    def _push_jar(self, jar: Path) -> None:
        # push only when the device copy differs (push costs ~100 ms)
        remote_size = self._adb(["shell", "stat", "-c", "%s", DEVICE_JAR],
                                check=False).stdout.decode("utf-8", "replace").strip()
        if remote_size.isdigit() and int(remote_size) == jar.stat().st_size:
            return
        self._adb(["push", str(jar), DEVICE_JAR], timeout=60)

    def start(self) -> "ScrcpyControl":
        """Start the server and connect the control socket."""
        jar = ensure_server_jar(self.jar)
        self._push_jar(jar)

        # a stale server would keep the abstract socket busy
        self._adb(["shell", "pkill", "-f", "com.genymobile.scrcpy.Server"], check=False)
        self._adb(["forward", "--remove", f"tcp:{self.port}"], check=False)

        options = [
            SCRCPY_VERSION,
            f"log_level={self.log_level}",
            "video=false",
            "audio=false",
            "control=true",
            "tunnel_forward=true",
            "cleanup=false",
            "power_on=false",
            "clipboard_autosync=false",
            f"stay_awake={str(self.keep_awake).lower()}",
            "send_dummy_byte=true",
            "send_device_meta=true",
        ]
        command = f"CLASSPATH={DEVICE_JAR} app_process / com.genymobile.scrcpy.Server " + " ".join(options)
        self.server = subprocess.Popen(
            self._adb_command(["shell", command]),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        self._adb(["forward", f"tcp:{self.port}", "localabstract:scrcpy"], timeout=30)

        deadline = time.monotonic() + self.timeout
        last_error: Optional[Exception] = None
        while time.monotonic() < deadline:
            if self.server.poll() is not None:
                detail = (self.server.stderr.read() or b"").decode("utf-8", "replace").strip()
                raise ScrcpyError(f"scrcpy server exited: {detail or self.server.returncode}")
            try:
                sock = socket.create_connection(("127.0.0.1", self.port), timeout=2.0)
                sock.settimeout(max(2.0, self.timeout))
                # one dummy byte, then the device name (padded to 64 bytes)
                dummy = sock.recv(1)
                if dummy != b"\x00":
                    sock.close()
                    raise ScrcpyError(f"unexpected dummy byte from server: {dummy!r}")
                name = b""
                while len(name) < 64:
                    chunk = sock.recv(64 - len(name))
                    if not chunk:
                        break
                    name += chunk
                self.device_name = name.rstrip(b"\x00").decode("utf-8", "replace")
                self.sock = sock
                return self
            except (OSError, ScrcpyError) as exc:
                last_error = exc
                time.sleep(0.1)

        self.close()
        raise ScrcpyError(f"could not connect to the scrcpy control socket: {last_error}")

    # --------------------------------------------------------------- control
    def _send(self, packet: bytes) -> None:
        if self.sock is None:
            raise ScrcpyError("control socket is not connected")
        try:
            self.sock.sendall(packet)
        except OSError as exc:
            raise ScrcpyError(f"control socket closed: {exc}") from exc

    def touch(self, action: int, pointer_id: int, x: float, y: float,
              screen_width: int = 1080, screen_height: int = 2392,
              pressure: float = 1.0) -> None:
        """Send one pointer event; the server merges pointers into one gesture."""
        value = int(round(max(0.0, min(1.0, pressure)) * 65535))
        self._send(struct.pack(
            ">BBqiiHHHII",
            MSG_TOUCH,
            action & 0xFF,
            int(pointer_id),
            int(round(x)),
            int(round(y)),
            int(screen_width) & 0xFFFF,
            int(screen_height) & 0xFFFF,
            value,
            0,                     # action_button
            0,                     # buttons
        ))

    def down(self, x: float, y: float, pointer_id: int = 1, **kwargs) -> None:
        self.touch(ACTION_DOWN, pointer_id, x, y, **kwargs)

    def move(self, x: float, y: float, pointer_id: int = 1, **kwargs) -> None:
        self.touch(ACTION_MOVE, pointer_id, x, y, **kwargs)

    def up(self, x: float, y: float, pointer_id: int = 1, **kwargs) -> None:
        self.touch(ACTION_UP, pointer_id, x, y, **kwargs)

    def tap(self, x: float, y: float, pointer_id: int = 0,
            screen_width: int = 1080, screen_height: int = 2392,
            hold_ms: float = 0.0) -> None:
        kwargs = dict(screen_width=screen_width, screen_height=screen_height)
        self.down(x, y, pointer_id, **kwargs)
        if hold_ms:
            time.sleep(hold_ms / 1000.0)
        self.up(x, y, pointer_id, **kwargs)

    def tap_many(self, points: Iterable[Tuple[float, float]], gap_ms: float = 20.0,
                 pointer_id: int = 0, screen_width: int = 1080,
                 screen_height: int = 2392, hold_ms: float = 0.0) -> int:
        """Tap every point in order; no adb process per tap."""
        gap = max(0.0, gap_ms) / 1000.0
        count = 0
        for x, y in points:
            self.tap(x, y, pointer_id, screen_width, screen_height, hold_ms=hold_ms)
            count += 1
            if gap:
                time.sleep(gap)
        return count

    def swipe(self, x0: float, y0: float, x1: float, y1: float,
              steps: int = 10, duration_ms: float = 250.0, pointer_id: int = 0,
              screen_width: int = 1080, screen_height: int = 2392) -> None:
        """One-finger drag (the game uses this to pan the camera)."""
        kwargs = dict(screen_width=screen_width, screen_height=screen_height)
        steps = max(1, int(steps))
        delay = max(0.0, duration_ms) / 1000.0 / steps
        self.down(x0, y0, pointer_id, **kwargs)
        try:
            for step in range(1, steps + 1):
                self.move(x0 + (x1 - x0) * step / steps,
                          y0 + (y1 - y0) * step / steps, pointer_id, **kwargs)
                if delay:
                    time.sleep(delay)
        finally:
            self.up(x1, y1, pointer_id, **kwargs)

    def pinch(self, center_x: float, center_y: float,
              start_gap: float, end_gap: float,
              steps: int = 8, duration_ms: float = 220.0,
              pointer_ids: Tuple[int, int] = (0, 1),
              angle: float = 0.0,
              screen_width: int = 1080, screen_height: int = 2392,
              settle_ms: float = 0.0) -> None:
        """Two-finger pinch; a shrinking gap zooms the game camera out.

        Pointer ids 0 and 1 are what Android apps expect for the first and second
        finger.  The UP events are sent at the final positions, not the initial
        ones, so gesture detectors that look at the release point agree too.
        """
        import math

        dx = math.cos(math.radians(angle))
        dy = math.sin(math.radians(angle))
        first, second = pointer_ids
        kwargs = dict(screen_width=screen_width, screen_height=screen_height)

        def point(gap: float, side: int) -> Tuple[float, float]:
            return (center_x + side * gap * dx / 2.0,
                    center_y + side * gap * dy / 2.0)

        steps = max(1, int(steps))
        delay = max(0.0, duration_ms) / 1000.0 / steps
        a_x, a_y = point(start_gap, -1)
        b_x, b_y = point(start_gap, +1)
        try:
            self.down(a_x, a_y, first, **kwargs)
            self.down(b_x, b_y, second, **kwargs)
            for step in range(1, steps + 1):
                gap = start_gap + (end_gap - start_gap) * step / steps
                a_x, a_y = point(gap, -1)
                b_x, b_y = point(gap, +1)
                self.move(a_x, a_y, first, **kwargs)
                self.move(b_x, b_y, second, **kwargs)
                if delay:
                    time.sleep(delay)
        finally:
            self.up(a_x, a_y, first, **kwargs)
            self.up(b_x, b_y, second, **kwargs)
        if settle_ms:
            time.sleep(settle_ms / 1000.0)

    def zoom_out(self, center_x: float, center_y: float, distance: float = 260.0,
                 target: float = 60.0, **kwargs) -> None:
        """Pinch two fingers together (the usual zoom-out gesture)."""
        self.pinch(center_x, center_y, distance, target, **kwargs)

    def zoom_in(self, center_x: float, center_y: float, distance: float = 60.0,
                target: float = 260.0, **kwargs) -> None:
        """Pinch two fingers apart."""
        self.pinch(center_x, center_y, distance, target, **kwargs)

    # --------------------------------------------------------------- cleanup
    def close(self) -> None:
        sock, self.sock = self.sock, None
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass
        if self.server is not None:
            self.server.terminate()
            try:
                self.server.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.server.kill()
            self.server = None
        try:
            self._adb(["forward", "--remove", f"tcp:{self.port}"], timeout=10, check=False)
        except Exception:
            pass

    def __enter__(self) -> "ScrcpyControl":
        return self.start()

    def __exit__(self, *exc_info) -> None:
        self.close()


class ScrcpyDevice:
    """The agent's device: screenshots over adb, touch over scrcpy.

    All the game needs is a screen and taps, so this class exposes exactly the
    same interface as :class:`arrowbot.adb.Adb` plus real multi-touch gestures.
    If scrcpy is unavailable the caller simply uses the Adb object instead.
    """

    def __init__(self, adb, control: ScrcpyControl,
                 screen_width: int = 1080, screen_height: int = 2392,
                 hold_ms: float = 0.0):
        self.adb = adb
        self.control = control
        self.screen_width = int(screen_width)
        self.screen_height = int(screen_height)
        self.hold_ms = float(hold_ms)

    @classmethod
    def open(cls, adb, config=None) -> "ScrcpyDevice":
        """Start the server and return a ready device."""
        control = ScrcpyControl(
            adb_path=getattr(adb, "adb_path", "adb"),
            serial=getattr(adb, "serial", None),
            jar=getattr(config, "scrcpy_jar", None),
            port=getattr(config, "scrcpy_port", 27183),
            log_level=getattr(config, "scrcpy_log_level", "error"),
        )
        control.start()
        try:
            width, height = adb.screen_size()
        except Exception:
            control.close()
            raise
        return cls(adb, control, width, height,
                   hold_ms=getattr(config, "tap_hold_ms", 0.0))

    # ---------------------------------------------------------------- screen
    def screencap(self):
        return self.adb.screencap()

    # ----------------------------------------------------------------- input
    def tap(self, x: float, y: float, hold_ms: Optional[float] = None) -> None:
        self.control.tap(x, y, screen_width=self.screen_width,
                         screen_height=self.screen_height,
                         hold_ms=self.hold_ms if hold_ms is None else hold_ms)

    def tap_many(self, points, gap_ms: float = 0.0,
                 hold_ms: Optional[float] = None) -> int:
        return self.control.tap_many(
            points, gap_ms=gap_ms, screen_width=self.screen_width,
            screen_height=self.screen_height,
            hold_ms=self.hold_ms if hold_ms is None else hold_ms)

    def swipe(self, x0: float, y0: float, x1: float, y1: float,
              steps: int = 10, duration_ms: float = 250.0) -> None:
        self.control.swipe(x0, y0, x1, y1, steps=steps, duration_ms=duration_ms,
                           screen_width=self.screen_width,
                           screen_height=self.screen_height)

    def pinch(self, center_x: float, center_y: float,
              start_gap: float, end_gap: float, steps: int = 8,
              duration_ms: float = 220.0, settle_ms: float = 0.0) -> None:
        self.control.pinch(center_x, center_y, start_gap, end_gap,
                           steps=steps, duration_ms=duration_ms,
                           screen_width=self.screen_width,
                           screen_height=self.screen_height,
                           settle_ms=settle_ms)

    # --------------------------------------------------------------- cleanup
    def close(self) -> None:
        self.control.close()

    def __enter__(self) -> "ScrcpyDevice":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()
