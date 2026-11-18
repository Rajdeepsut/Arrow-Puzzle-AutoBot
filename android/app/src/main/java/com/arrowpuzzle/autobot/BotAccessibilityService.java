package com.arrowpuzzle.autobot;

import android.content.Context;
import android.graphics.Rect;
import android.os.Build;
import android.os.Handler;
import android.os.Looper;
import android.os.PowerManager;
import android.os.SystemClock;
import android.provider.Settings;
import android.view.WindowManager;
import android.view.accessibility.AccessibilityEvent;
import android.view.accessibility.AccessibilityNodeInfo;
import android.accessibilityservice.AccessibilityService;
import android.accessibilityservice.GestureDescription;
import android.graphics.Path;

import java.util.Locale;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;

/**
 * Performs taps and pinches for the bot and tracks which app is in front.
 *
 * <p>Only gestures are used to touch the screen; overlay buttons are located
 * through the accessibility tree and then tapped at their centre.  The bot only
 * calls {@link #clickOverlay} when it cannot see a board.</p>
 */
public class BotAccessibilityService extends AccessibilityService {
    private static volatile BotAccessibilityService instance;

    private final Handler mainHandler = new Handler(Looper.getMainLooper());
    private volatile String foregroundPackage = "";
    private volatile long lastEventAt;
    private volatile boolean cachedForeground;
    private volatile long cachedAt;
    private volatile long lastRootAt;

    public static BotAccessibilityService getInstance() {
        return instance;
    }

    public static boolean isEnabled(Context context) {
        String enabled = Settings.Secure.getString(context.getContentResolver(),
                Settings.Secure.ENABLED_ACCESSIBILITY_SERVICES);
        if (enabled == null) {
            return false;
        }
        String full = context.getPackageName() + "/" + BotAccessibilityService.class.getName();
        String shortName = context.getPackageName() + "/.BotAccessibilityService";
        return enabled.contains(full) || enabled.contains(shortName);
    }

    public static boolean isConnected() {
        return instance != null;
    }

    @Override
    protected void onServiceConnected() {
        super.onServiceConnected();
        instance = this;
    }

    @Override
    public void onAccessibilityEvent(AccessibilityEvent event) {
        if (event == null) {
            return;
        }
        CharSequence name = event.getPackageName();
        if (name != null && name.length() > 0) {
            foregroundPackage = name.toString();
            lastEventAt = SystemClock.elapsedRealtime();
        }
    }

    @Override
    public void onInterrupt() {
    }

    @Override
    public boolean onUnbind(android.content.Intent intent) {
        instance = null;
        return super.onUnbind(intent);
    }

    // ------------------------------------------------------------ foreground
    public boolean isGameForeground() {
        long now = SystemClock.elapsedRealtime();
        if (now - cachedAt < 250) {
            return cachedForeground;
        }
        cachedAt = now;

        boolean result = AppConfig.GAME_PACKAGE.equals(foregroundPackage);
        // WINDOW_STATE_CHANGED is authoritative, but some launchers and system
        // dialogs do not deliver one.  Check the active window occasionally too.
        if (now - lastRootAt >= 1000 || !result) {
            lastRootAt = now;
            AccessibilityNodeInfo root = null;
            try {
                root = getRootInActiveWindow();
                if (root != null && root.getPackageName() != null) {
                    result = AppConfig.GAME_PACKAGE.contentEquals(root.getPackageName());
                }
            } catch (Exception ignored) {
                // keep the event-based answer
            } finally {
                if (root != null && Build.VERSION.SDK_INT < 33) {
                    root.recycle();
                }
            }
        }
        cachedForeground = result;
        return result;
    }

    public static boolean screenOn(Context context) {
        PowerManager power = (PowerManager) context.getSystemService(Context.POWER_SERVICE);
        return power != null && power.isInteractive();
    }

    // ----------------------------------------------------------------- input
    public boolean tapBatch(int[] xs, int[] ys, int count, int gapMs, int holdMs) {
        if (count <= 0 || count > GestureDescription.getMaxStrokeCount()
                || xs == null || ys == null || xs.length < count || ys.length < count) {
            return false;
        }
        int[] size = screenSize();
        GestureDescription.Builder builder = new GestureDescription.Builder();
        long gap = Math.max(0, gapMs);
        long hold = Math.max(1, holdMs);
        for (int i = 0; i < count; i++) {
            if (xs[i] < 0 || ys[i] < 0 || xs[i] >= size[0] || ys[i] >= size[1]) {
                return false;
            }
            Path path = new Path();
            path.moveTo(xs[i], ys[i]);
            long start = i * gap;
            try {
                builder.addStroke(new GestureDescription.StrokeDescription(path, start, hold));
            } catch (IllegalArgumentException exc) {
                return false;
            }
        }
        long timeout = 2500 + count * (gap + hold);
        return dispatch(builder.build(), timeout);
    }

    public boolean pinch(int cx, int cy, int startGap, int endGap, int durationMs) {
        int[] size = screenSize();
        if (cx < 0 || cy < 0 || cx >= size[0] || cy >= size[1]) {
            return false;
        }
        int leftStart = Math.max(0, cx - startGap / 2);
        int rightStart = Math.min(size[0] - 1, cx + startGap / 2);
        int leftEnd = Math.max(0, cx - endGap / 2);
        int rightEnd = Math.min(size[0] - 1, cx + endGap / 2);
        long duration = Math.max(1, durationMs);

        GestureDescription.Builder builder = new GestureDescription.Builder();
        Path left = new Path();
        left.moveTo(leftStart, cy);
        left.lineTo(leftEnd, cy);
        Path right = new Path();
        right.moveTo(rightStart, cy);
        right.lineTo(rightEnd, cy);
        try {
            builder.addStroke(new GestureDescription.StrokeDescription(left, 0, duration));
            builder.addStroke(new GestureDescription.StrokeDescription(right, 0, duration));
        } catch (IllegalArgumentException exc) {
            return false;
        }
        return dispatch(builder.build(), duration + 2500);
    }

    private boolean dispatch(GestureDescription gesture, long timeoutMs) {
        CountDownLatch latch = new CountDownLatch(1);
        boolean[] accepted = {false};
        boolean[] completed = {false};
        Runnable dispatch = () -> {
            try {
                accepted[0] = dispatchGesture(gesture,
                        new GestureResultCallback() {
                            @Override
                            public void onCompleted(GestureDescription gestureDescription) {
                                completed[0] = true;
                                latch.countDown();
                            }

                            @Override
                            public void onCancelled(GestureDescription gestureDescription) {
                                latch.countDown();
                            }
                        }, null);
            } catch (Exception ignored) {
                accepted[0] = false;
            }
            if (!accepted[0]) {
                latch.countDown();
            }
        };
        if (Looper.myLooper() == Looper.getMainLooper()) {
            dispatch.run();
        } else {
            mainHandler.post(dispatch);
        }
        try {
            return latch.await(timeoutMs, TimeUnit.MILLISECONDS) && accepted[0] && completed[0];
        } catch (InterruptedException exc) {
            Thread.currentThread().interrupt();
            return false;
        }
    }

    private int[] screenSize() {
        WindowManager manager = (WindowManager) getSystemService(WINDOW_SERVICE);
        if (Build.VERSION.SDK_INT >= 30) {
            Rect bounds = manager.getMaximumWindowMetrics().getBounds();
            return new int[]{bounds.width(), bounds.height()};
        }
        android.util.DisplayMetrics metrics = new android.util.DisplayMetrics();
        manager.getDefaultDisplay().getRealMetrics(metrics);
        return new int[]{metrics.widthPixels, metrics.heightPixels};
    }

    // --------------------------------------------------------------- overlays
    public boolean clickOverlay(String[] labels) {
        AccessibilityNodeInfo root = null;
        try {
            root = getRootInActiveWindow();
            if (root == null || root.getPackageName() == null
                    || !AppConfig.GAME_PACKAGE.contentEquals(root.getPackageName())) {
                return false;
            }
            Candidate best = new Candidate();
            walk(root, labels, 0, new Rect(), best);
            if (best.score < 0 || best.bounds.isEmpty()) {
                return false;
            }
            int x = best.bounds.centerX();
            int y = best.bounds.centerY();
            return tapBatch(new int[]{x}, new int[]{y}, 1, 0, 40);
        } catch (Exception ignored) {
            return false;
        } finally {
            if (root != null && Build.VERSION.SDK_INT < 33) {
                root.recycle();
            }
        }
    }

    private void walk(AccessibilityNodeInfo node, String[] labels, int depth,
                      Rect clickable, Candidate best) {
        if (node == null || depth > 30) {
            return;
        }
        Rect bounds = new Rect();
        node.getBoundsInScreen(bounds);
        Rect inherited = clickable;
        if (node.isClickable() && !bounds.isEmpty()) {
            inherited = bounds;
        }
        String text = lower(node.getText());
        String description = lower(node.getContentDescription());
        if (node.isVisibleToUser() && !bounds.isEmpty()) {
            for (int i = 0; i < labels.length; i++) {
                String label = labels[i];
                int match = matchText(text, label);
                if (match < 0) {
                    match = matchText(description, label);
                }
                if (match >= 0) {
                    int score = i * 100 + match;
                    if (best.score < 0 || score < best.score) {
                        best.score = score;
                        best.bounds.set(inherited.isEmpty() ? bounds : inherited);
                    }
                    break;
                }
            }
        }
        for (int i = 0; i < node.getChildCount(); i++) {
            AccessibilityNodeInfo child = null;
            try {
                child = node.getChild(i);
                walk(child, labels, depth + 1, inherited, best);
            } catch (Exception ignored) {
                // stale node: skip this branch
            } finally {
                if (child != null && Build.VERSION.SDK_INT < 33) {
                    child.recycle();
                }
            }
        }
    }

    private static int matchText(String value, String label) {
        if (value == null || value.isEmpty() || label == null || label.isEmpty()) {
            return -1;
        }
        if (value.equals(label)) {
            return 0;
        }
        if (label.length() >= 4 && value.startsWith(label)) {
            return 5;
        }
        if (label.length() >= 5 && value.contains(label)) {
            return 10;
        }
        return -1;
    }

    private static String lower(CharSequence text) {
        return text == null ? "" : text.toString().toLowerCase(Locale.US).trim();
    }

    private static final class Candidate {
        int score = -1;
        final Rect bounds = new Rect();
    }
}
