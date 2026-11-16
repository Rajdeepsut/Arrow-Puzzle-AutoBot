# Arrow Puzzle Bot

[![CI](https://github.com/Rajdeepsut/Arrow-Puzzle-AutoBot/actions/workflows/ci.yml/badge.svg)](https://github.com/Rajdeepsut/Arrow-Puzzle-AutoBot/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

HELLO DEv My self Rajdeep Sut i developed something cute .An autonomous agent that plays **Arrow Puzzle: Tap Puzzle Games** (Easybrain) on a
real Android phone: it screenshots the board with ADB, reads every arrow with
OpenCV, works out which arrows can leave the board without hitting anything, taps
them, and repeats until the level is empty - then taps through to the next level
and keeps going.

No device-specific assets, no template images, no manual grid calibration
required. Grid size (4x4 up to 9x9 and beyond), cell size, colours and the
light/dark theme are all detected per frame.

```
        screenshot (adb exec-out screencap -p)
                    |
                    v
   +-----------------------------------+
   | vision.py                         |
   |  blob detection -> lattice fit    |
   |  -> 4-way orientation classifier  |
   +-----------------------------------+
                    |  {(row, col): "U"|"D"|"L"|"R"}
                    v
   +-----------------------------------+
   | solver.py   ray casting           |
   |  safe = no arrow in front         |
   +-----------------------------------+
                    |  list of arrows to tap
                    v
        adb shell input tap x y   (repeat)
```

## Why the greedy strategy is provably correct

An arrow is **safe** when no other arrow sits strictly in front of it, along its
own direction, up to the edge of the board. Tapping a safe arrow only *removes*
an arrow, and removing an arrow can never put an obstacle into somebody else's
lane. So the set of safe arrows can only grow as the board empties:

* the order in which safe arrows are tapped can never hurt, and
* if the greedy sweep gets stuck with arrows left over, **no** tap order could
  have cleared that position (it is dead-locked, not unlucky).

The solver therefore needs no search, no backtracking and no lookahead: one
sweep per wave, and the loop terminates.

## Requirements

* Python 3.9 - 3.13 (CI runs the whole suite on every version)
* `numpy` and `opencv-python` (`pip install -r requirements.txt`)
* Android platform-tools (`adb`) on your PATH, USB debugging enabled
* The game open on a level

## Install

Pick one:

```bash
# a) run it from the checkout
pip install -r requirements.txt

# b) install it as a command (`arrowbot ...` instead of `python main.py ...`)
pip install -e .
```

Then verify the install - no phone needed:

```bash
python main.py selftest          # or: arrowbot selftest
```

`selftest` runs 43 checks: the game rules, the vision pipeline against synthetic
screens (5x5..8x8, half-empty boards, dark theme, 1 arrow left) and the full
agent loop against a simulated phone - including a check that the bot **never
taps a blocked arrow**.

## Quick start

```bash
# 1. is the phone visible?
python main.py devices

# 2. look at what the bot sees, and save an annotated overlay
python main.py shot

# 3. plan only, tap nothing
python main.py run --dry-run

# 4. play: 500 levels, starting from wherever the game currently is
python main.py run --max-levels 500
```

`python main.py shot` prints the parsed board, e.g.

```
28 arrows | mask=dark-on-light spacing=100.0px origin=(10.4,89.7) candidates=28 dropped=0

L L U U U U
L L R . . U
L L D U . D
U . D D . D
D . D . . L
D D D L D R
```

and writes `debug/shot.png` with every arrow circled, labelled with its direction,
confidence and the fitted grid. **Always check this image first** - if the board
looks right there, the run will be right.

You can analyse a saved screenshot without a phone:

```bash
python main.py shot --image samples/sample_board.png
```

## Playing

```bash
python main.py run --serial 1234567890 --max-levels 500 --log-file debug/run.log
```

Each level:

1. wait until the screen stops changing (skips launch animations),
2. detect the arrows and classify their directions,
3. tap every safe arrow, `chunk_size` at a time, re-reading between chunks,
4. stop tapping the moment the board is empty,
5. wait for the level-complete screen, tap through it, confirm the next board.

Debug images for anything unusual (deadlock, stalled, popup) land in `debug/`.

## Configuration

`config.json` is optional - every key has a default (shown below). Create the
file with only the keys you want to change; unknown keys are ignored.

| key | default | meaning |
| --- | --- | --- |
| `adb_path` | `"adb"` | path to the adb binary |
| `serial` | `null` | device serial; auto-detected when only one is attached |
| `roi` | `[0, 0.10, 1, 0.90]` | region searched for arrows, as screen fractions `(x0, y0, x1, y1)` |
| `exclude_rects` | `[]` | extra ignored rectangles (ad banners, HUD), same fraction format |
| `min_blob_area_frac` | `0.00004` | smallest glyph, fraction of the screen area |
| `max_blob_area_frac` | `0.06` | largest glyph; only has to reject big UI blocks |
| `morph_close` | `3` | kernel that closes gaps inside a glyph (0 = off) |
| `spacing_tolerance` | `0.22` | how far a blob may sit off the fitted grid, in cells |
| `min_orientation_confidence` | `0.6` | arrows read less confidently are never tapped |
| `tap_delay_ms` | `140` | pause between two taps of a wave |
| `chunk_size` | `3` | taps between two screen reads (1 = safest, slowest) |
| `settle_timeout_ms` | `3500` | max wait for the screen to stop animating |
| `settle_poll_ms` | `120` | screenshot interval while waiting |
| `settle_thresh` | `2.0` | mean grey difference that still counts as "same frame" |
| `tap_retries` | `2` | retries when a wave has no visible effect |
| `max_waves_per_level` | `400` | hard stop for one level |
| `advance_attempts` | `6` | taps allowed to get past the level-complete screen |
| `advance_wait_ms` | `900` | wait between those taps |
| `next_tap_points` | `[[0.5,0.86],[0.5,0.80],[0.5,0.91]]` | where the "next level" button is, as screen fractions |
| `on_deadlock` | `"stop"` | `stop` or `tap_anyway` (sacrifices a life and continues) |
| `continue_on_failure` | `false` | keep going after a level that did not finish |
| `max_levels` | `500` | how many levels to clear in this run |
| `start_level` | `1` | label used in logs only (`config.json` ships `32`) |
| `stop_on_error` | `false` | re-raise exceptions instead of stopping cleanly |
| `write_debug` | `true` | write annotated images to `debug_dir` |
| `debug_dir` | `"debug"` | where those images go |

### Calibration

Ninety percent of setups work with the defaults. If `python main.py shot` shows a
wrong or partial board:

1. **Missing arrows at the top/bottom** - widen `roi`. The default skips the top
   and bottom 10% of the screen, which is where the HUD and the buttons live.
2. **A banner or ad is being read as arrows** - add it to `exclude_rects`, e.g.
   `"exclude_rects": [[0.0, 0.93, 1.0, 1.0]]`.
3. **`origin`/`spacing` look wrong, or arrows are merged** - look at the grid
   lines in `debug/shot.png`. If they do not run between the arrows, the glyphs
   are probably touching; raise `morph_close` to `5` or lower
   `spacing_tolerance`.
4. **Directions are misread (`conf` well below 1.00)** - lower
   `min_orientation_confidence` slightly (0.5) or raise it (0.8) to make the bot
   refuse those arrows instead. Confidence is the agreement of three independent
   geometric votes.
5. **The bot does not get to the next level** - it is tapping the wrong place.
   Point `next_tap_points` at the actual button (screenshot fractions).

## Speed

One level costs roughly `avl arrows / chunk_size` screenshot+detect cycles. On a
1080p phone a cycle is ~0.4-0.8 s, so a 36-arrow level takes ~6-10 s and 500
levels about an hour. Knobs:

* `chunk_size` 3 -> 5 speeds things up; the risk is that one tap of a chunk lands
  on the level-complete screen.
* `tap_delay_ms` 140 -> 80 if your phone registers fast taps reliably.
* `settle_poll_ms` 120 -> 80 to react a bit quicker.

Keep `chunk_size` low if you care more about never wasting a life.

## Troubleshooting

| symptom | cause / fix |
| --- | --- |
| `no adb devices` | enable USB debugging, accept the RSA prompt, try another cable/port, `adb kill-server` |
| `adb: device unauthorized` | accept the debugging prompt on the phone |
| `no arrows found` | game not at a level, wrong `roi`, or an ad is covering the board |
| taps land in the wrong place | the phone is showing the game in a resized window/one-handed mode; make it full screen |
| it stops with `deadlock` | a glyph was misread (check `debug/L*_deadlock.png`) or the level really is stuck |
| it stops with `changed` | a popup/ad appeared mid-level; close it and rerun |
| it stops with `stalled` | taps had no effect twice; usually a dialog is waiting |

Nothing in this bot depends on the level number, so you can start it on any level
the game shows and it will just play.

## Limitations (read this)

* It was developed and verified against **synthetic** screens and a simulated
  phone, not against your specific build of the game. The algorithm is
  game-agnostic, but UI details (ROI, next-button position) may need the one-time
  calibration above.
* **Ads and popups need a human.** The bot detects that something new appeared
  and stops instead of tapping blindly.
* Levels that use special tiles (locked arrows, double arrows, bombs, ice) are
  not modelled - such a tile will be read as an ordinary arrow, and the bot will
  refuse to tap it if the glyph is ambiguous, which shows up as a deadlock you
  can inspect.
* Screen capture is parsed as a flat grid; rotation, split screen and in-game
  zoom are not handled.
* Automating a game can violate its terms of service. Use this on your own device
  and your own account, at your own risk.

## Project layout

```
main.py                 CLI: run / shot / solve / devices / selftest
config.json             the defaults above, editable
pyproject.toml          packaging + pytest config (installs the `arrowbot` command)
arrowbot/
  solver.py             pure ray-casting rules and planner (no dependencies)
  vision.py             masks, blob filter, lattice fit, orientation voting
  agent.py              the level loop, tap scheduling, debug dumps
  adb.py                screencap / tap / device management
  sim.py                synthetic phone used by the self-test
  selftest.py           43 end-to-end checks, no phone needed
tests/test_solver.py    30 unit tests for the rules (pytest)
samples/sample_board.png  a synthetic screenshot for `shot --image`
.github/workflows/ci.yml  lint + both test suites on Python 3.9 / 3.11 / 3.12 / 3.13
```

## Tests

```bash
ruff check .             # lint: dead code, undefined names, import order
python -m pytest -q      # rules only, milliseconds
python main.py selftest  # rules + vision + full agent loop, a few seconds
```

Lint and both suites run in CI on every push, across Python 3.9 / 3.11 / 3.12 /
3.13, with no phone and no display attached.

## License

MIT - see [LICENSE](LICENSE).
