# Arrow AutoBot - the standalone APK

This folder is a second, self-contained way to run the bot: instead of a PC
driving the phone over ADB/scrcpy, an Android app on the phone reads its own
screen with `MediaProjection`, runs the same vision/solver/counter logic in
Java, and performs taps through an `AccessibilityService`. No root, no PC, no
ADB, no Termux.

The PC bot in the repository root remains the reference implementation. The
Java core here is a line-for-line port of it, and
`app/src/test/java/.../VisionRegressionTest.java` fails the build if it ever
reads the three real screenshots differently from the Python pipeline.

## Requirements

* Android 8.0 (API 26) or newer; built and tested against API 35
* The game installed: `com.easybrain.arrow.puzzle.game`
* Accessibility permission (for gestures) and one screen-capture consent per
  app run
* A phone whose manufacturer does not kill background work - see
  [vivo / OriginOS](#vivo--originos)

## Install

The APK is built by `.github/workflows/android.yml` on every push that touches
`android/**`:

1. open the **android** workflow run on GitHub,
2. download the `arrowbot-debug-apk` artifact,
3. copy `app-debug.apk` to the phone and tap it (allow "install unknown apps"
   for the file manager once).

The debug APK is signed with the standard Android debug key, so it installs
without any extra steps. Android will warn that the app was not scanned by Play
Protect; that warning is expected for a sideloaded debug build.

To build it yourself you need JDK 17 and an Android SDK with platform 35:

```bash
cd android
gradle :app:testDebugUnitTest :app:assembleDebug
# app/build/outputs/apk/debug/app-debug.apk
```

The unit test runs on a plain JVM and needs no emulator or phone.

## First run

1. Open **Arrow AutoBot** and tap **Open settings** next to *Accessibility*.
2. Enable **Arrow AutoBot gestures** under *Installed services* / *Downloaded
   apps*. It only performs gestures and reads the active window's package name.
3. Optional but recommended: tap **Fix** next to *Battery* and allow
   unrestricted battery use. On vivo/OriginOS also enable auto-start and
   background activity.
4. Open Arrow Puzzle and leave it on a level.
5. Return to Arrow AutoBot, press **Start bot**, accept screen capture, and let
   it open the game (the *Open Arrow Puzzle after starting* box).

Once running, the app shows a notification with **Stop**. The bot keeps playing
across levels; it does not need a button press between boards.

Starting the bot while already inside a level works: it waits for two matching
frames, reads that board and plays it.

## What it does on the phone

The loop is the same as the PC bot:

1. capture the screen through `MediaProjection` (`ImageReader`, RGBA -> ARGB),
2. crop the board region `[0, 0.14, 1, 0.79]`, build the ink mask, run an
   exact Euclidean distance transform, take one maximum per thick head, split
   touching arrows between heads, and match a synthetic triangle for direction,
3. read the HUD remaining-arrow counter and the hearts as a second opinion,
4. plan the complete greedy clear order - every arrow in it is provably safe in
   that order,
5. send the first tap alone, then batches of up to 10 arrows in one
   `dispatchGesture` call (Android's hard limit is 10 strokes per gesture),
6. screenshot between batches: counter mismatch, a lost heart, moved heads, an
   arrow near the screen edge or a clipped board immediately drops the batch to
   one tap per screenshot,
7. repeat until the board is empty, then wait for the next board.

No tap is ever guessed. If the board cannot be read, the bot waits, dismisses a
`Continue`/`Retry`-style overlay found in the accessibility tree, or re-reads -
it does not tap a board it does not understand.

## Timing

The PC bot clears real 56/74/78-arrow boards in a few seconds of tapping; the
APK is bounded by on-device capture and the game's removal animation rather
than by input speed. The defaults target well under 60 s per level, and the
usual case is roughly 10 s:

| stage | default |
| --- | --- |
| settle wait between captures | 70 ms poll, 2.5 s timeout |
| first batch | 1 tap (probe) |
| later batches | 6, growing to 10 while the camera is still |
| gap / press inside a batch | 35 ms / 25 ms |
| pause after a batch | 180 ms |
| single-tap fallback wait | 280 ms |

All of these live in
`app/src/main/java/com/arrowpuzzle/autobot/core/Config.java`; change the
constants there and rebuild. The safety gates are not tunable into a guess:
batching only grows while surviving heads move less than `batchDriftPx`.

## Troubleshooting

| symptom | fix |
| --- | --- |
| *Accessibility is off* even after enabling | some ROMs kill the service; re-open the app and check *Installed services* |
| capture stops as soon as the screen turns off | set battery use to *Unrestricted*, enable auto-start, raise the screen timeout |
| *Screen capture was not allowed* | consent is per session on Android 14+; press Start again and choose the whole display |
| bot sits on *Waiting for a level* | the game is not in the foreground, an ad is covering the board, or the board region is wrong |
| bot keeps saying *Cautious mode* | counter/hearts disagree with the reading; it is playing safely, one tap per screenshot |
| taps land slightly off | the phone is in a resized/one-handed window; run the game full screen |
| *Could not start screen capture* | a previous projection was not released; force-stop Arrow AutoBot and start again |

## Tests and regenerated data

```bash
# Java core vs. the Python pipeline on the three real captures
python tools/export_android_reference.py   # only after shared logic changes
python tools/export_android_templates.py   # only after counter templates change
cd android && gradle :app:testDebugUnitTest
```

`reference.tsv` and `DigitTemplates.java` are generated files; do not edit them
by hand. The test expects 208 arrows across `real_level.png`,
`real_level_74.png` and `real_level_zoomed.png`, with exact head coordinates,
directions, scales, bounding boxes, HUD values and complete safe plans. The
frames themselves are stored as zlib-compressed raw pixels (`*.argb.z`) so the
JUnit test does not need AWT, which is absent from the Android unit-test boot
classpath.

## Layout

```
android/
  app/src/main/java/com/arrowpuzzle/autobot/
    core/            pure Java: Config, Detector, Solver, Counter, Player
    MainActivity     setup screen and live status
    BotService       foreground service, MediaProjection lifecycle
    ScreenCapture    ImageReader -> ARGB Frame
    BotAccessibilityService  taps, pinches, foreground tracking, overlay clicks
    AndroidTransport BotTransport over the two services
  app/src/test/      JUnit regression against the real screenshots
  app/src/test/resources/reference.tsv   Python-generated expected readings
```

## Caveats

* Special tiles (locked/double arrows, bombs, ice) are not modelled, exactly as
  in the PC bot; the bot re-reads, repairs a direction, or slows down instead of
  guessing.
* Automating a game can violate its terms of service. Use it on your own device
  and account, at your own risk.
