package com.arrowpuzzle.autobot.core;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * The level loop.  Reads the board, taps only provably safe arrows, checks the
 * HUD counter and hearts, and keeps going board after board until stopped.
 *
 * <p>Safety invariant: every tap comes from a prefix of a valid greedy clear
 * order, and that prefix is re-checked against the live ink before it is sent.
 * If the board cannot be read, is clipped, or dead-locks, the bot waits or
 * falls back to one tap per screenshot - it never guesses and never blindly
 * taps the board.</p>
 */
public final class Player {
    private static final String[] OVERLAY_LABELS = {
            "continue", "resume", "retry", "play", "next", "ok", "close",
            "no thanks", "claim", "skip", "restart",
    };

    private final Config config;
    private final Detector detector;
    private final BotTransport transport;
    private final BotListener listener;

    private volatile boolean running;
    private volatile String phase = "Idle";
    private volatile String detail = "";
    private volatile int levelsCleared;
    private volatile int totalTaps;
    private volatile int totalBatches;
    private volatile double lastLevelSeconds;
    private volatile int lastLevelArrows;

    private int heartsSeen;
    private int tapsThisLevel;
    private long lastOverlayAt;
    private boolean zoomInward = true;

    public Player(Config config, BotTransport transport, BotListener listener) {
        this.config = config;
        this.detector = new Detector(config);
        this.transport = transport;
        this.listener = listener;
    }

    public void run() {
        running = true;
        update("Starting", "Waiting for the game");
        while (running) {
            if (!transport.isScreenOn() || !transport.isGameForeground()) {
                update("Waiting for Arrow Puzzle", "Open the game to start");
                sleep(config.idlePollMs);
                continue;
            }
            Detection detection = readStable();
            if (!running) {
                break;
            }
            if (detection == null) {
                maybeClickOverlay();
                update("Waiting for a stable board", "The camera is still moving");
                sleep(120);
                continue;
            }
            if (!looksLikeBoard(detection)) {
                maybeClickOverlay();
                update("Waiting for a level", "No arrows on screen");
                sleep(200);
                continue;
            }

            update("Playing level", detection.describe());
            listener.onBoard(detection, totalTaps);
            int arrows = detection.arrows.size();
            long started = System.nanoTime();
            tapsThisLevel = 0;
            String outcome = playBoard(detection);
            if (!running) {
                break;
            }
            if ("cleared".equals(outcome)) {
                double seconds = (System.nanoTime() - started) / 1_000_000_000.0;
                levelsCleared++;
                lastLevelSeconds = seconds;
                lastLevelArrows = arrows;
                listener.onLevelCleared(seconds, arrows, tapsThisLevel);
                update("Level cleared", String.format(java.util.Locale.US,
                        "%.1f s, %d arrows, %d taps", seconds, arrows, tapsThisLevel));
                sleep(250);
            } else if ("paused".equals(outcome)) {
                update("Paused", "Game left the foreground");
                sleep(300);
            } else {
                listener.onFailure(outcome);
                update("Retrying", hint(outcome));
                sleep(600);
            }
        }
        update("Stopped", "");
    }

    public void stop() {
        running = false;
    }

    public boolean isRunning() {
        return running;
    }

    public String phase() {
        return phase;
    }

    public String detail() {
        return detail;
    }

    public int levelsCleared() {
        return levelsCleared;
    }

    public int totalTaps() {
        return totalTaps;
    }

    public int totalBatches() {
        return totalBatches;
    }

    public double lastLevelSeconds() {
        return lastLevelSeconds;
    }

    public int lastLevelArrows() {
        return lastLevelArrows;
    }

    // ------------------------------------------------------------------ reads
    private Detection detectOnce() {
        for (int attempt = 0; attempt < 2; attempt++) {
            try {
                Frame frame = transport.capture(1200);
                if (frame != null) {
                    return detector.detect(frame);
                }
            } catch (Exception ignored) {
                // a dropped frame is not fatal; try once more
            }
            sleep(70);
        }
        return null;
    }

    private Detection readStable() {
        long deadline = System.nanoTime()
                + config.settleTimeoutMs * 1_000_000L;
        Detection previous = null;
        while (running && System.nanoTime() < deadline) {
            Detection detection = detectOnce();
            if (detection != null) {
                if (previous != null
                        && detection.sameView(previous, config.stabilityTolPx)) {
                    return transport.isGameForeground() ? detection : null;
                }
                previous = detection;
            }
            sleep(config.settlePollMs);
        }
        return null;
    }

    private Detection probe() {
        Detection detection = detectOnce();
        return detection;
    }

    // ------------------------------------------------------------------ board
    private String playBoard(Detection initial) {
        heartsSeen = 0;
        List<Detection> seen = new ArrayList<>();
        Detection detection = ensureZoom(initial);
        if (detection == null) {
            return "unstable";
        }
        if (detection.arrows.isEmpty()) {
            return "empty";
        }
        if (!checkHearts(detection.frame)) {
            // A heart was already gone before the first tap.  That is not a
            // reason to stop the whole run, but it is a reason to be careful.
            update("Cautious mode", "A heart was already missing");
        }
        CounterResult counter = settleCounter(detection);
        detection = counter.detection;
        if (!counter.ok) {
            return playSlow(detection, "HUD count disagrees");
        }
        if (!allTappable(detection)) {
            return playSlow(detection, "uncertain or edge arrows");
        }
        if (!transport.isGameForeground()) {
            return "paused";
        }

        int previous = detection.arrows.size();
        int batchLimit = 1;          // the first tap of a level is always a probe
        int stalls = 0;
        int waves = 0;
        while (running && !detection.arrows.isEmpty()) {
            if (!transport.isGameForeground() || !transport.isScreenOn()) {
                return "paused";
            }
            if (waves++ >= config.maxWavesPerLevel) {
                return "exhausted";
            }
            remember(seen, detection);

            if (boardClipped(detection)) {
                Detection fitted = ensureZoom(detection);
                if (fitted == null) {
                    return "unstable";
                }
                detection = fitted;
                if (boardClipped(detection)) {
                    return playSlow(detection, "board clipped at the edge");
                }
                if (!allTappable(detection)) {
                    return playSlow(detection, "arrows at the screen edge");
                }
                previous = detection.arrows.size();
            }

            List<Arrow> order = Solver.order(detection.arrows);
            if (order == null) {
                Map<Arrow, Character> fixes = Solver.repair(detection.arrows);
                if (fixes == null || fixes.isEmpty()) {
                    update("Deadlock", "No safe tap; re-reading the board");
                    return "deadlock";
                }
                for (Map.Entry<Arrow, Character> fix : fixes.entrySet()) {
                    fix.getKey().direction = fix.getValue();
                }
                order = Solver.order(detection.arrows);
                if (order == null) {
                    return "deadlock";
                }
                update("Repaired a direction", "Re-read the ambiguous arrow");
            }

            int count = Math.min(batchLimit, order.size());
            if (!safePrefix(order, count, detection.arrows, detection.frame)) {
                update("Unsafe plan blocked", "Re-reading the board");
                return "deadlock";
            }
            Detection before = detection;
            int[] xs = new int[count];
            int[] ys = new int[count];
            for (int i = 0; i < count; i++) {
                xs[i] = order.get(i).headX;
                ys[i] = order.get(i).headY;
            }
            update("Tapping", count + " of " + detection.arrows.size()
                    + " arrows (batch " + (totalBatches + 1) + ")");
            boolean sent = transport.tapBatch(xs, ys, count,
                    config.batchGapMs, config.tapHoldMs);
            tapsThisLevel += count;
            totalTaps += count;
            totalBatches++;
            sleep(config.batchWaitMs);

            Detection after = probe();
            if (after == null) {
                return "error";
            }
            detection = after;
            CounterResult settled = settleCounter(detection);
            detection = settled.detection;
            if (!settled.ok) {
                batchLimit = 1;
            }
            boolean heartsOk = checkHearts(detection.frame);
            if (!heartsOk) {
                batchLimit = 1;
            }

            int afterCount = detection.arrows.size();
            if (afterCount > previous) {
                if (sameAsSeenExceptLast(seen, detection)) {
                    previous = afterCount;
                    continue;
                }
                if (previous <= count) {
                    // the batch covered every arrow: the level cleared and the
                    // next board may already be on screen
                    return "cleared";
                }
                return "changed";
            }
            if (afterCount == previous) {
                sleep(config.batchWaitMs);
                Detection retry = probe();
                if (retry != null) {
                    detection = retry;
                    CounterResult second = settleCounter(detection);
                    detection = second.detection;
                    if (!second.ok) {
                        batchLimit = 1;
                    }
                    heartsOk = checkHearts(detection.frame);
                    if (!heartsOk) {
                        batchLimit = 1;
                    }
                    afterCount = detection.arrows.size();
                }
                if (afterCount == previous) {
                    stalls++;
                    if (stalls > config.tapRetries) {
                        return "stalled";
                    }
                } else {
                    stalls = 0;
                }
            } else {
                stalls = 0;
                if (afterCount > previous - count) {
                    // the removal animation was still running when we looked
                    sleep(config.batchWaitMs);
                    Detection more = probe();
                    if (more != null && more.arrows.size() <= afterCount) {
                        detection = more;
                    }
                }
            }
            previous = detection.arrows.size();
            if (!sent) {
                batchLimit = 1;
            }
            if (detection.arrows.isEmpty()) {
                continue;
            }

            if (!checkHearts(detection.frame)) {
                batchLimit = 1;
            }
            if (!allTappable(detection)) {
                return playSlow(detection, "an arrow reached the screen edge");
            }
            float drift = cameraDrift(before, detection);
            if (!Float.isNaN(drift)) {
                if (drift > config.batchDriftPx) {
                    batchLimit = drift > 2 * config.batchDriftPx
                            ? 1 : Math.max(1, batchLimit / 2);
                } else {
                    batchLimit = Math.min(config.batchSize,
                            Math.max(config.batchSizeStart, batchLimit * 2));
                }
            }
        }
        return "cleared";
    }

    /** One tap per screenshot: the safest fallback, used when anything is uncertain. */
    private String playSlow(Detection initial, String reason) {
        update("Cautious mode", reason);
        Detection detection = initial;
        List<Detection> seen = new ArrayList<>();
        int previous = -1;
        int stalls = 0;
        int steps = 0;
        while (running && detection != null && !detection.arrows.isEmpty()) {
            if (!transport.isGameForeground() || !transport.isScreenOn()) {
                return "paused";
            }
            if (steps++ >= config.maxWavesPerLevel) {
                return "exhausted";
            }
            remember(seen, detection);
            int count = detection.arrows.size();
            if (previous >= 0) {
                if (count > previous) {
                    if (sameAsSeenExceptLast(seen, detection)) {
                        previous = count;
                        continue;
                    }
                    return "changed";
                }
                if (count == previous) {
                    // The removal animation can be slower than one poll; look
                    // again before calling this tap a stall.
                    sleep(config.tapWaitMs);
                    Detection fresh = probe();
                    if (fresh != null) {
                        detection = fresh;
                        CounterResult settled = settleCounter(detection);
                        detection = settled.detection;
                        if (detection.arrows.size() < count) {
                            previous = detection.arrows.size();
                            stalls = 0;
                            continue;
                        }
                    }
                    stalls++;
                    if (stalls > config.tapRetries) {
                        return "stalled";
                    }
                } else {
                    stalls = 0;
                }
            }
            checkHearts(detection.frame);

            List<Arrow> safe = candidates(detection);
            if (safe.isEmpty()) {
                Map<Arrow, Character> fixes = Solver.repair(detection.arrows);
                if (fixes == null || fixes.isEmpty()) {
                    return "deadlock";
                }
                for (Map.Entry<Arrow, Character> fix : fixes.entrySet()) {
                    fix.getKey().direction = fix.getValue();
                }
                safe = candidates(detection);
                if (safe.isEmpty()) {
                    return "deadlock";
                }
                update("Repaired a direction", "Trying one arrow");
            }
            Arrow pick = safe.get(0);
            if (!Solver.isSafe(pick, detection.arrows)
                    || pick.headX < 0 || pick.headX >= detection.frame.width
                    || pick.headY < 0 || pick.headY >= detection.frame.height) {
                return "deadlock";
            }
            update("Tapping one arrow", pick.toString());
            boolean sent = transport.tapBatch(new int[]{pick.headX}, new int[]{pick.headY}, 1,
                    config.batchGapMs, config.tapHoldMs);
            tapsThisLevel++;
            totalTaps++;
            totalBatches++;
            if (!sent) {
                sleep(120);
            }
            sleep(config.tapWaitMs);
            Detection after = probe();
            if (after == null) {
                return "error";
            }
            detection = after;
            CounterResult settled = settleCounter(detection);
            detection = settled.detection;
            checkHearts(detection.frame);
            previous = count;
        }
        return detection != null && detection.arrows.isEmpty() ? "cleared" : "empty";
    }

    // ------------------------------------------------------------------ checks
    private CounterResult settleCounter(Detection detection) {
        if (!config.counterCheck || detection == null) {
            return new CounterResult(detection, true);
        }
        for (int attempt = 0; attempt <= config.counterRetries; attempt++) {
            Integer hud = Counter.read(detection.frame, config);
            if (hud == null || hud == detection.arrows.size()) {
                return new CounterResult(detection, true);
            }
            if (attempt >= config.counterRetries) {
                return new CounterResult(detection, false);
            }
            sleep(config.counterWaitMs);
            Detection fresh = probe();
            if (fresh != null) {
                detection = fresh;
            }
        }
        return new CounterResult(detection, false);
    }

    private boolean checkHearts(Frame frame) {
        int red = redPixels(frame);
        if (red < config.heartsMinRed) {
            return true;             // hidden HUD (banner/ad) - cannot tell
        }
        if (heartsSeen > 0 && red < 0.85f * heartsSeen) {
            heartsSeen = red;
            update("Heart lost", "Continuing with one tap per screenshot");
            return false;
        }
        heartsSeen = Math.max(heartsSeen, red);
        return true;
    }

    private int redPixels(Frame frame) {
        int[] box = Detector.fractionsToBox(config.heartsRoi, frame.width, frame.height);
        int count = 0;
        for (int y = box[1]; y < box[3]; y++) {
            int row = y * frame.width;
            for (int x = box[0]; x < box[2]; x++) {
                int pixel = frame.pixels[row + x];
                int r = (pixel >>> 16) & 0xff;
                if (r > 140 && ((pixel >>> 8) & 0xff) < 110 && (pixel & 0xff) < 110) {
                    count++;
                }
            }
        }
        return count;
    }

    private List<Arrow> candidates(Detection detection) {
        int edge = config.edgeMarginPx;
        int width = detection.frame.width;
        List<Arrow> confident = new ArrayList<>();
        List<Arrow> nearEdge = new ArrayList<>();
        for (Arrow arrow : detection.arrows) {
            if (arrow.confidence < config.minDirectionMargin) {
                continue;
            }
            if (arrow.headX < edge || arrow.headX >= width - edge) {
                nearEdge.add(arrow);
            } else {
                confident.add(arrow);
            }
        }
        List<Arrow> pool = confident.isEmpty() ? nearEdge : confident;
        List<Arrow> safe = new ArrayList<>();
        for (Arrow arrow : pool) {
            if (Solver.isSafe(arrow, detection.arrows)) {
                safe.add(arrow);
            }
        }
        safe.sort((a, b) -> Float.compare(b.confidence, a.confidence));
        return safe;
    }

    private boolean allTappable(Detection detection) {
        int edge = config.edgeMarginPx;
        int width = detection.frame.width;
        for (Arrow arrow : detection.arrows) {
            if (arrow.confidence < config.minDirectionMargin) {
                return false;
            }
            if (arrow.headX < edge || arrow.headX >= width - edge) {
                return false;
            }
        }
        return true;
    }

    private boolean looksLikeBoard(Detection detection) {
        if (detection == null || detection.arrows.isEmpty()) {
            return false;
        }
        if (detection.arrows.size() >= 4) {
            return true;
        }
        if (redPixels(detection.frame) >= config.heartsMinRed) {
            return true;
        }
        Integer hud = Counter.read(detection.frame, config);
        return hud != null && hud == detection.arrows.size();
    }

    private boolean safePrefix(List<Arrow> order, int count, List<Arrow> alive, Frame frame) {
        List<Arrow> remaining = new ArrayList<>(alive);
        for (int i = 0; i < count; i++) {
            Arrow arrow = order.get(i);
            if (arrow.headX < 0 || arrow.headX >= frame.width
                    || arrow.headY < 0 || arrow.headY >= frame.height) {
                return false;
            }
            if (!Solver.isSafe(arrow, remaining)) {
                return false;
            }
            remaining.remove(arrow);
        }
        return true;
    }

    private float cameraDrift(Detection before, Detection after) {
        if (before.arrows.isEmpty() || after.arrows.isEmpty()) {
            return Float.NaN;
        }
        float radius = 24f * Math.max(0.5f, after.scale);
        boolean[] used = new boolean[after.arrows.size()];
        float worst = 0f;
        int measured = 0;
        for (Arrow old : before.arrows) {
            int bestIndex = -1;
            float best = radius;
            for (int i = 0; i < after.arrows.size(); i++) {
                if (used[i]) {
                    continue;
                }
                Arrow fresh = after.arrows.get(i);
                float dx = old.headX - fresh.headX;
                float dy = old.headY - fresh.headY;
                float distance = (float) Math.hypot(dx, dy);
                if (distance < best) {
                    best = distance;
                    bestIndex = i;
                }
            }
            if (bestIndex >= 0) {
                used[bestIndex] = true;
                worst = Math.max(worst, best);
                measured++;
            }
        }
        return measured >= 3 ? worst : Float.NaN;
    }

    // -------------------------------------------------------------------- zoom
    private Detection ensureZoom(Detection detection) {
        if (detection == null || !config.autoZoom || !boardClipped(detection)) {
            return detection;
        }
        Detection latest = detection;
        for (int attempt = 0; attempt < config.zoomAttempts; attempt++) {
            if (!boardClipped(latest)) {
                return latest;
            }
            int centerX = Math.round(config.zoomCenterX * latest.frame.width);
            int centerY = Math.round(config.zoomCenterY * latest.frame.height);
            int start = zoomInward ? config.zoomStartGapPx : config.zoomEndGapPx;
            int end = zoomInward ? config.zoomEndGapPx : config.zoomStartGapPx;
            update("Zooming out", "The board is clipped at the edge");
            if (!transport.pinch(centerX, centerY, start, end, config.zoomDurationMs)) {
                break;
            }
            sleep(config.zoomSettleMs);
            Detection after = readStable();
            if (after == null) {
                continue;
            }
            boolean better = betterFit(after, latest);
            latest = after;
            if (better) {
                continue;
            }
            zoomInward = !zoomInward;
        }
        return latest;
    }

    private boolean betterFit(Detection first, Detection second) {
        boolean firstFits = !boardClipped(first);
        boolean secondFits = !boardClipped(second);
        if (firstFits != secondFits) {
            return firstFits;
        }
        if (first.arrows.size() != second.arrows.size()) {
            return first.arrows.size() > second.arrows.size();
        }
        return first.scale < second.scale;
    }

    private boolean boardClipped(Detection detection) {
        if (detection.arrows.isEmpty()) {
            return false;
        }
        int[] roi = Detector.fractionsToBox(config.roi,
                detection.frame.width, detection.frame.height);
        int margin = config.zoomEdgePx;
        return detection.x0 <= roi[0] + margin
                || detection.y0 <= roi[1] + margin
                || detection.x1 >= roi[2] - 1 - margin
                || detection.y1 >= roi[3] - 1 - margin;
    }

    // ---------------------------------------------------------------- overlays
    private void maybeClickOverlay() {
        long now = System.nanoTime() / 1_000_000L;
        if (now - lastOverlayAt < config.overlayPollMs) {
            return;
        }
        lastOverlayAt = now;
        if (transport.clickOverlay(OVERLAY_LABELS)) {
            update("Dismissed an overlay", "Looking for the next board");
            sleep(350);
        }
    }

    private void remember(List<Detection> seen, Detection detection) {
        if (detection.arrows.isEmpty()) {
            return;
        }
        seen.add(detection);
        while (seen.size() > 4) {
            seen.remove(0);
        }
    }

    private boolean sameAsSeenExceptLast(List<Detection> seen, Detection detection) {
        for (int i = 0; i < seen.size() - 1; i++) {
            if (detection.sameView(seen.get(i), 3)) {
                return true;
            }
        }
        return false;
    }

    private void update(String phase, String detail) {
        this.phase = phase;
        this.detail = detail;
        listener.onPhase(phase, detail);
    }

    private void sleep(long millis) {
        if (millis <= 0) {
            return;
        }
        try {
            Thread.sleep(millis);
        } catch (InterruptedException exc) {
            Thread.currentThread().interrupt();
            running = false;
        }
    }

    private static String hint(String outcome) {
        switch (outcome) {
            case "empty":
                return "No arrows on screen - waiting for a level";
            case "unstable":
                return "The camera never stopped moving";
            case "changed":
                return "Something new covered the board";
            case "stalled":
                return "Taps had no visible effect - re-reading";
            case "deadlock":
                return "No safe arrow right now - re-reading";
            case "life_lost":
                return "A heart was lost";
            case "exhausted":
                return "Stopped after too many batches";
            case "error":
                return "Screen capture hiccup - retrying";
            case "paused":
                return "Game left the foreground";
            default:
                return "Re-reading the board";
        }
    }

    private static final class CounterResult {
        final Detection detection;
        final boolean ok;

        CounterResult(Detection detection, boolean ok) {
            this.detection = detection;
            this.ok = ok;
        }
    }
}
