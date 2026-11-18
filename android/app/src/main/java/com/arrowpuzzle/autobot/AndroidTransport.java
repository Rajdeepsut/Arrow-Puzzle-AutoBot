package com.arrowpuzzle.autobot;

import android.content.Context;
import android.os.PowerManager;

import com.arrowpuzzle.autobot.core.BotTransport;
import com.arrowpuzzle.autobot.core.Frame;

/** The Android side of the level loop: capture, gestures, foreground state. */
final class AndroidTransport implements BotTransport {
    private final ScreenCapture capture;
    private final PowerManager power;

    AndroidTransport(Context context, ScreenCapture capture) {
        this.capture = capture;
        this.power = (PowerManager) context.getSystemService(Context.POWER_SERVICE);
    }

    @Override
    public Frame capture(int timeoutMs) {
        return capture.capture(timeoutMs);
    }

    @Override
    public boolean tapBatch(int[] xs, int[] ys, int count, int gapMs, int holdMs) {
        BotAccessibilityService service = BotAccessibilityService.getInstance();
        return service != null && service.tapBatch(xs, ys, count, gapMs, holdMs);
    }

    @Override
    public boolean pinch(int cx, int cy, int startGap, int endGap, int durationMs) {
        BotAccessibilityService service = BotAccessibilityService.getInstance();
        return service != null && service.pinch(cx, cy, startGap, endGap, durationMs);
    }

    @Override
    public boolean isGameForeground() {
        BotAccessibilityService service = BotAccessibilityService.getInstance();
        return service != null && service.isGameForeground();
    }

    @Override
    public boolean isScreenOn() {
        return power != null && power.isInteractive();
    }

    @Override
    public boolean clickOverlay(String[] labels) {
        BotAccessibilityService service = BotAccessibilityService.getInstance();
        return service != null && service.clickOverlay(labels);
    }
}
