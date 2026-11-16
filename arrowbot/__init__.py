"""Autonomous solver for the Android game "Arrow Puzzle" (Easybrain).

Modules
-------
config   : dataclass configuration, JSON load/save
solver   : pure ray-casting solver (no dependencies, unit-testable)
vision   : screen -> board model (OpenCV)
adb      : thin ADB wrapper (screencap / tap)
agent    : the level loop that ties everything together
sim      : synthetic "phone" used by the self-test
selftest : end-to-end verification without a real device
"""

__version__ = "1.0.0"
