package com.arrowpuzzle.autobot.core;

/**
 * Every tunable the on-device bot needs, with the values measured on the real
 * 1080x2392 phone.  Kept deliberately small: the APK has no JSON config file.
 */
public final class Config {
    // ------------------------------------------------------------------ vision
    /** Region that may contain the board, as screen fractions (x0,y0,x1,y1). */
    public float[] roi = {0.0f, 0.14f, 1.0f, 0.79f};
    /** Grey value below which a pixel counts as navy arrow ink. */
    public int inkGrayMax = 110;
    /** Red arrows are ordinary arrows and must not be missed. */
    public boolean redInk = true;
    /** Artwork scale accepted by the detector. */
    public float scaleMin = 0.5f;
    public float scaleMax = 4.0f;
    /** Minimum template margin before an arrow may be tapped. */
    public float minDirectionMargin = 0.06f;
    /** An arrow with fewer ink pixels than this (at scale 1) is noise. */
    public int minArrowPixels = 60;
    /** Fixed minimum head size in distance-transform pixels; 0 = automatic. */
    public int minHeadPixels = 0;

    // ----------------------------------------------------------------- counter
    public boolean counterCheck = true;
    public float[] counterRoi = {0.02f, 0.09f, 0.20f, 0.126f};
    public int counterRetries = 2;
    public int counterWaitMs = 150;
    public float counterMinScore = 0.82f;

    // ------------------------------------------------------------------ hearts
    public float[] heartsRoi = {0.31f, 0.095f, 0.72f, 0.132f};
    public int heartsMinRed = 300;

    // ----------------------------------------------------------------- tapping
    /**
     * Arrows per batch.  Android's GestureDescription allows at most ten
     * strokes per gesture, so ten is also the hard ceiling.
     */
    public int batchSize = 10;
    /** First multi-tap batch; grows towards batchSize while the camera is still. */
    public int batchSizeStart = 6;
    /** Surviving heads moving more than this between batches mean the camera moved. */
    public int batchDriftPx = 3;
    /** Pause between two taps inside one gesture. */
    public int batchGapMs = 35;
    /** How long each finger stays down for one arrow. */
    public int tapHoldMs = 25;
    /** Pause after a batch, to let the removal animation finish. */
    public int batchWaitMs = 180;
    /** Screenshot interval while waiting for the camera to stop moving. */
    public int settlePollMs = 70;
    /** Give up waiting for a stable board after this long. */
    public int settleTimeoutMs = 2500;
    /** Two frames are the same view when every head moved less than this. */
    public int stabilityTolPx = 2;
    /** Taps without any visible effect before giving up on this attempt. */
    public int tapRetries = 2;
    /** Never tap closer than this to the screen edge (clipped ink misleads). */
    public int edgeMarginPx = 18;
    /** Hard safety valve for one board, not a normal limit. */
    public int maxWavesPerLevel = 400;

    // -------------------------------------------------------------------- zoom
    public boolean autoZoom = true;
    public float zoomCenterX = 0.5f;
    public float zoomCenterY = 0.45f;
    public int zoomStartGapPx = 280;
    public int zoomEndGapPx = 70;
    public int zoomSteps = 8;
    public int zoomDurationMs = 180;
    public int zoomAttempts = 3;
    public int zoomSettleMs = 300;
    public int zoomEdgePx = 3;

    // ------------------------------------------------------------------ timing
    /** Pause after a single tap before the board is read again. */
    public int tapWaitMs = 280;
    /** How often the loop looks for the game when it is not foreground. */
    public int idlePollMs = 300;
    /** Overlay button clicks are throttled to this interval. */
    public int overlayPollMs = 900;

    public Config validate() {
        roi = fractionBox(roi, new float[]{0f, 0f, 1f, 1f});
        counterRoi = fractionBox(counterRoi, new float[]{0.02f, 0.09f, 0.20f, 0.126f});
        heartsRoi = fractionBox(heartsRoi, new float[]{0.31f, 0.095f, 0.72f, 0.132f});
        inkGrayMax = clamp(inkGrayMax, 1, 250);
        minDirectionMargin = clamp(minDirectionMargin, 0f, 0.9f);
        minArrowPixels = Math.max(8, minArrowPixels);
        minHeadPixels = Math.max(0, minHeadPixels);
        scaleMin = Math.max(0.1f, scaleMin);
        scaleMax = Math.max(scaleMin * 1.5f, scaleMax);
        counterRetries = Math.max(0, counterRetries);
        counterWaitMs = Math.max(0, counterWaitMs);
        counterMinScore = clamp(counterMinScore, 0f, 1f);
        heartsMinRed = Math.max(0, heartsMinRed);
        batchSize = clamp(batchSize, 1, 10);
        batchSizeStart = clamp(batchSizeStart, 1, batchSize);
        batchDriftPx = Math.max(0, batchDriftPx);
        batchGapMs = Math.max(0, batchGapMs);
        tapHoldMs = Math.max(1, tapHoldMs);
        batchWaitMs = Math.max(0, batchWaitMs);
        settlePollMs = Math.max(10, settlePollMs);
        settleTimeoutMs = Math.max(settlePollMs * 2, settleTimeoutMs);
        stabilityTolPx = Math.max(0, stabilityTolPx);
        tapRetries = Math.max(0, tapRetries);
        edgeMarginPx = Math.max(0, edgeMarginPx);
        maxWavesPerLevel = Math.max(1, maxWavesPerLevel);
        zoomStartGapPx = Math.max(20, zoomStartGapPx);
        zoomEndGapPx = Math.max(10, zoomEndGapPx);
        zoomSteps = Math.max(1, zoomSteps);
        zoomDurationMs = Math.max(0, zoomDurationMs);
        zoomAttempts = Math.max(0, zoomAttempts);
        zoomSettleMs = Math.max(0, zoomSettleMs);
        zoomEdgePx = Math.max(0, zoomEdgePx);
        tapWaitMs = Math.max(0, tapWaitMs);
        idlePollMs = Math.max(50, idlePollMs);
        overlayPollMs = Math.max(200, overlayPollMs);
        return this;
    }

    static float[] fractionBox(float[] values, float[] fallback) {
        if (values == null || values.length != 4) {
            return fallback.clone();
        }
        float x0 = clamp(values[0], 0f, 1f);
        float y0 = clamp(values[1], 0f, 1f);
        float x1 = clamp(values[2], 0f, 1f);
        float y1 = clamp(values[3], 0f, 1f);
        if (x1 - x0 < 0.01f) {
            x0 = 0f;
            x1 = 1f;
        }
        if (y1 - y0 < 0.01f) {
            y0 = 0f;
            y1 = 1f;
        }
        return new float[]{x0, y0, x1, y1};
    }

    static int clamp(int value, int low, int high) {
        return Math.max(low, Math.min(high, value));
    }

    static float clamp(float value, float low, float high) {
        return Math.max(low, Math.min(high, value));
    }
}
