package com.arrowpuzzle.autobot;

/** Volatile state shared by the service and the activity (no AndroidX). */
public final class BotStatus {
    public static volatile boolean running;
    public static volatile String phase = "Idle";
    public static volatile String detail = "";
    public static volatile int levels;
    public static volatile int taps;
    public static volatile int batches;
    public static volatile int lastArrows;
    public static volatile double lastSeconds;
    public static volatile int captureWidth;
    public static volatile int captureHeight;
    public static volatile String failure = "";

    private BotStatus() {
    }

    public static void reset() {
        running = false;
        phase = "Idle";
        detail = "";
        captureWidth = 0;
        captureHeight = 0;
        failure = "";
    }
}
