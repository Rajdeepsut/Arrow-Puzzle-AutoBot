# Arrow Puzzle Bot

[![CI](https://github.com/Rajdeepsut/Arrow-Puzzle-AutoBot/actions/workflows/ci.yml/badge.svg)](https://github.com/Rajdeepsut/Arrow-Puzzle-AutoBot/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

> HELLO DEv My self Rajdeep Sut i developed something cute .

An autonomous agent that plays **Arrow Puzzle: Tap Puzzle Games** (Easybrain) on a
real Android phone. It screenshots the board with ADB, finds every arrow with
OpenCV, works out which arrows can leave without crossing another arrow's ink,
taps them in batches over a persistent scrcpy socket, and moves on to the next
level by itself.

The target is **one level per minute** - in practice, a level takes a few
seconds of tapping once the page has loaded.

```
     screenshot (adb exec-out screencap -p)
                    |
                    v
  +------------------------------------------+
  | vision.py                                |
  |  ink mask -> distance-transform heads    |
  |  -> 4-way triangle templates (rotated)   |
  +------------------------------------------+
                    |  Arrow(head, direction, ink pixels, scale)
                    v
  +------------------------------------------+
  | solver.py                                |
  |  safe = nothing in the corridor ahead    |
  +------------------------------------------+
                    |  full clear order (greedy waves)
                    v
  +------------------------------------------+
  | agent.py                                 |
  |  zoom-out pinch -> fast batch taps       |
  |  -> verify hearts/count -> repeat        |
  +------------------------------------------+
                    |  scrcpy touch messages (multi-touch, no root)
                    v
                 the phone
```

## Why greedy is provably correct

An arrow is **safe** when no other arrow's ink lies in the corridor in front of
its head: the half-plane it points into, restricted to a narrow band around its
own axis. Tapping a safe arrow only *removes* ink, and removing an arrow can
never put an obstacle into somebody else's corridor. So:

* the order of safe taps can never hurt, and
* if the greedy sweep gets stuck with arrows left, **no** tap order could have
  cleared that position - it is a genuine deadlock (or a misread arrow).

There is no search, no backtracking, and no guessing.

## Why batch tapping is safe

Because removal never creates a blocker, the complete greedy order can be
computed once and executed as-is. Measurements on the real game at its default
camera fit showed that after a tap **all 73 remaining arrow heads moved 0.0 px**
and the artwork scale stayed at 1.60, so coordinates stay valid while a batch is
being tapped. The bot still takes a screenshot between batches (6 arrows at
first, growing to 8 while the camera stands still) to catch a panned or zoomed
camera, a lost heart, a stalled tap, or a popup. The simulator is harsher than
the real game - it zooms in on every removal - and this batching scheme clears
it without ever losing a heart.

## Zoom-out at level start

The game remembers the camera zoom, so a new level can open partially off-screen.
The first step of every level is therefore:

1. take a stable screenshot;
2. check whether any arrow ink touches the edge of the board region;
3. if it does, perform a two-finger pinch (scrcpy pointer ids 0 and 1) and
   re-check; both pinch directions are tried and the one that actually shrinks
   the artwork is remembered for later levels.

When the board already fits - the normal case - **no gesture is sent**, so the
step costs nothing. `--no-zoom` disables it.

## Requirements

* Python 3.9 - 3.13
* `numpy`, `opencv-python` (`pip install -r requirements.txt`)
* Android platform-tools (`adb`) on PATH, USB debugging enabled
* Java is **not** needed on the PC: the small scrcpy server jar is downloaded
  once (SHA-256 verified) and pushed to the phone. scrcpy itself does not need to
  be installed.
* The game open on a level

## Install

```bash
pip install -r requirements.txt
# or, to get an `arrowbot` command:
pip install -e .
```

Verify the install - no phone needed:

```bash
python main.py selftest
```

`selftest` runs 51 checks: the corridor rules, detection against synthetic
screens of several sizes (navy, red, zoomed), the zoom-out logic, the full fast
agent loop (never loses a heart, advance to the next level, under a minute per
level), the scrcpy 32-byte packet format, and a moving board that must never be
tapped.

## Quick start

```bash
# 1. is the phone visible?
python main.py devices

# 2. what does the bot see?  saves an annotated overlay
python main.py shot

# 3. detect + zoom + plan, tap nothing
python main.py run --dry-run

# 4. play fast until 500 levels are cleared
python main.py run --max-levels 500 --start-level 33
```

Play on the current level (level 33 is a full 74-arrow board at the default
camera fit) with per-level timing in the log:

```bash
python main.py run --serial 10BF3S1JD900379 --start-level 33
```

A run logs, per level:

```
--- level 33 (0/500 cleared) ---
pinching together (zoom out): 280 -> 70 px at (540, 1077)   # only if clipped
fast mode: 74 arrows, batches of up to 8 (start 6, ~10 screenshots)
level 33 cleared in 4.2 s (74 taps, 17.6 arrows/s)
level cleared -> 1 total, 4.2 s, 74 taps
next level loaded (81 arrows)
```

If anything unusual happens, an annotated image is written to `debug/`
(`deadlock`, `life_lost`, `changed`, `stalled`).

## Commands

| command | what it does |
| --- | --- |
| `python main.py selftest` | rules + vision + zoom + fast agent, no phone |
| `python main.py devices` | list ADB devices |
| `python main.py shot [--image file.png]` | detect one frame and save `debug/shot.png` |
| `python main.py run` | play levels: fast mode, scrcpy input, auto zoom-out |
| `python main.py run --slow` | one tap per screenshot (moving camera) |
| `python main.py run --backend adb` | no scrcpy: `adb shell input tap`, no pinch |
| `python main.py run --dry-run` | detect, zoom and plan only |
| `python main.py run --batch-size 14` | more taps per screenshot (faster) |
| `python main.py run --no-zoom` | skip the level-start pinch |

## How a level runs (fast mode)

1. wait for a stable first frame (two identical readings);
2. zoom out if any arrow ink touches the board-region edge;
3. read the whole board, plan the complete clear order once;
4. tap the next batch (6 arrows, growing to 8 while the camera stands still) over
   one persistent scrcpy socket
   (down/up per arrow, ~6 ms apart, ~12 ms press);
5. take one screenshot: check hearts, count, and that no new arrows appeared;
6. repeat from step 3 until the board is empty;
7. wait for the next board and start again - no button press needed.

The game advances automatically when the board empties. If a batch contained
every remaining arrow and the next level has already loaded by the time the
verification screenshot arrives, the bot still records the level as cleared.

## Configuration

`config.json` is optional - every key has a default. Only listed keys are read;
unknown keys are ignored, so old files keep working.

| key | default | meaning |
| --- | --- | --- |
| `adb_path` | `"adb"` | path to the adb binary |
| `serial` | `null` | device serial; auto-detected when only one is attached |
| `input_backend` | `"scrcpy"` | `scrcpy` (fast, multi-touch) or `adb` |
| `scrcpy_jar` | `null` | optional path to a server jar; downloaded+verified otherwise |
| `scrcpy_port` | `27183` | local port forwarded to the scrcpy control socket |
| `roi` | `[0, 0.14, 1, 0.79]` | board region as screen fractions `(x0, y0, x1, y1)` |
| `exclude_rects` | `[]` | extra ignored rectangles, same format |
| `ink_gray_max` | `110` | grey level below which navy ink is detected |
| `red_ink` | `true` | also detect red arrows |
| `scale_range` | `[0.5, 4.0]` | accepted artwork scales (the game's camera zoom) |
| `min_direction_margin` | `0.06` | template margin below which a direction is ambiguous |
| `fast_mode` | `true` | plan once, tap batches |
| `batch_size` | `8` | maximum arrows per screenshot in fast mode |
| `batch_size_start` | `6` | first batch of a level (more conservative) |
| `adaptive_batches` | `true` | grow the batch back to `batch_size` while the camera is still |
| `batch_drift_px` | `3` | surviving-arrow movement that counts as "the camera moved" |
| `batch_gap_ms` | `6` | pause between taps of one batch |
| `tap_hold_ms` | `12` | press duration per arrow |
| `batch_wait_ms` | `160` | pause after a batch for the removal animation |
| `auto_zoom` | `true` | pinch out at level start if the board is clipped |
| `zoom_start_gap_px` / `zoom_end_gap_px` | `280` / `70` | pinch finger separation |
| `zoom_steps` / `zoom_duration_ms` | `8` / `180` | pinch smoothness |
| `zoom_attempts` | `3` | pinches allowed per level |
| `zoom_center` | `[0.5, 0.45]` | pinch centre, screen fractions |
| `chunk_size` | `1` | taps between reads in slow mode |
| `tap_wait_ms` | `700` | pause after a slow-mode tap |
| `settle_poll_ms` / `settle_timeout_ms` | `140` / `5000` | stable-view polling |
| `stability_tol_px` | `2` | maximum head movement that still counts as the same view |
| `tap_retries` | `2` | batched/individual taps without effect before giving up |
| `hearts_roi` | `[0.31, 0.095, 0.72, 0.132]` | strip that shows the hearts |
| `hearts_min_red` | `300` | fewer red pixels means the HUD is hidden |
| `edge_margin_px` | `18` | arrows closer to the screen edge fall back to slow mode |
| `max_waves_per_level` | `400` | safety valve for one level |
| `advance_timeout_ms` | `20000` | max wait for the next board |
| `on_deadlock` | `"stop"` | `stop` or `tap_anyway` |
| `continue_on_failure` | `false` | keep going after a failed level |
| `max_levels` | `500` | levels to clear in this run |
| `start_level` | `33` | log label for the first level |
| `write_debug` / `debug_dir` | `true` / `"debug"` | failure snapshots |

## Speed tuning

Measured with production defaults on the simulator: **~1.5-2.0 s per level**
(includes two screenshots per verify plus 160 ms animation waits and the odd
mid-level re-zoom). On a phone the screenshot dominates; a 74-arrow level needs
about 10 verification screenshots. Knobs, in order of impact:

* `batch_size` 8 -> 12-16: fewer screenshots, same provable tap order. Only do
  this if `--dry-run`/the log shows the camera stands still between batches; the
  bot re-checks between batches and shrinks back automatically when it does not.
* `batch_wait_ms` 160 -> 100: faster if your phone's removal animation is short.
* `batch_gap_ms` 6 -> 0 and `tap_hold_ms` 12 -> 6: raw input speed.
* `--slow` is the safe fallback, not the fast path.

## Troubleshooting

| symptom | cause / fix |
| --- | --- |
| `no adb devices` | enable USB debugging, accept the RSA prompt, `adb kill-server` |
| `device unauthorized` | accept the debugging prompt on the phone |
| `scrcpy is unavailable` | the jar download failed; set `scrcpy_jar` to a local copy or use `--backend adb` |
| `no arrows found` | game not on a level, wrong `roi`, or an ad is covering the board |
| board looks clipped in `debug/shot.png` | run with `--no-zoom` and pinch out manually, or raise `zoom_attempts` |
| taps land in the wrong place | phone is in a resized/one-handed window; use full screen |
| `deadlock` | a glyph was misread (check the image) or the level really is stuck |
| `changed` | a popup/ad appeared mid-level; close it and rerun |
| `stalled` | taps had no effect twice; usually a dialog is waiting |
| `life_lost` | a blocked arrow was tapped; inspect `debug/L*_life_lost.png` |

## Limitations

* Ads and popups need a human: the bot notices that the board changed and stops
  rather than tapping blindly.
* Levels with special tiles (locked arrows, double arrows, bombs, ice) are not
  modelled; an ambiguous glyph becomes a deadlock you can inspect.
* The two-finger pinch requires the scrcpy backend. With `--backend adb` the bot
  can still play a board that is already fully visible.
* Automating a game can violate its terms of service. Use it on your own device
  and account, at your own risk.

## Project layout

```
main.py                   CLI: run / shot / devices / selftest
config.json               editable defaults
arrowbot/
  solver.py               corridor geometry and greedy planner
  vision.py               ink mask, arrow heads, direction templates
  agent.py                zoom-out, fast batches, heart/stall checks
  adb.py                  screenshots, adb fallback input, device management
  scrcpy.py               scrcpy server lifecycle + multi-touch protocol
  sim.py                  synthetic phone (boards, camera zoom, hearts)
  selftest.py             51 end-to-end checks, no phone needed
tests/                    pytest: solver, vision/zoom, scrcpy packets
samples/sample_board.png  synthetic screenshot for `shot --image`
.github/workflows/ci.yml  lint + tests + selftest on Python 3.9/3.11/3.12/3.13
```

## Tests

```bash
ruff check .
python -m pytest -q
python main.py selftest
python main.py shot --image samples/sample_board.png
```

## License

MIT - see [LICENSE](LICENSE).
