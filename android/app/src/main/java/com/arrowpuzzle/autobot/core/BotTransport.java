package com.arrowpuzzle.autobot.core;

/** Everything the level loop needs from Android (capture + gestures + foreground state). */
public interface BotTransport {
    /**
     * Latest screen capture.  The returned frame is reused by the next call,
     * so callers must detect on it before capturing again.
     */
    Frame capture(int timeoutMs) throws Exception;

    /** Tap several points, in order, as one gesture. */
    boolean tapBatch(int[] xs, int[] ys, int count, int gapMs, int holdMs);

    /** Two-finger pinch around (cx,cy); gap is the distance between fingers. */
    boolean pinch(int cx, int cy, int startGap, int endGap, int durationMs);

    boolean isGameForeground();

    boolean isScreenOn();

    /** Press a visible overlay button by label; only called when no board is seen. */
    boolean clickOverlay(String[] labels);
}
