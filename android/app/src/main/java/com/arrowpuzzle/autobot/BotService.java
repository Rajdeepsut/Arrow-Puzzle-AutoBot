package com.arrowpuzzle.autobot;

import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.app.Service;
import android.content.Context;
import android.content.Intent;
import android.content.pm.ServiceInfo;
import android.graphics.drawable.Icon;
import android.media.projection.MediaProjection;
import android.media.projection.MediaProjectionManager;
import android.os.Build;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.IBinder;
import android.os.Process;
import android.os.SystemClock;

import com.arrowpuzzle.autobot.core.BotListener;
import com.arrowpuzzle.autobot.core.Config;
import com.arrowpuzzle.autobot.core.Detection;
import com.arrowpuzzle.autobot.core.Player;

/** Foreground service that owns MediaProjection and the level loop. */
public class BotService extends Service {
    public static final String ACTION_START = "com.arrowpuzzle.autobot.action.START";
    public static final String ACTION_STOP = "com.arrowpuzzle.autobot.action.STOP";
    public static final String EXTRA_RESULT_CODE = "resultCode";
    public static final String EXTRA_DATA = "resultData";

    private static final String CHANNEL_ID = "arrowbot";
    private static final int NOTIFICATION_ID = 42;

    private final Handler mainHandler = new Handler();
    private HandlerThread worker;
    private Handler workerHandler;
    private Player player;
    private ScreenCapture capture;
    private MediaProjection projection;
    private volatile boolean stopping;
    private boolean foregroundStarted;
    private long lastNotificationAt;
    private String lastNotificationText = "";

    @Override
    public void onCreate() {
        super.onCreate();
        NotificationChannel channel = new NotificationChannel(CHANNEL_ID,
                getString(R.string.notification_channel),
                NotificationManager.IMPORTANCE_LOW);
        channel.setDescription(getString(R.string.notification_channel_description));
        channel.setShowBadge(false);
        NotificationManager manager = getSystemService(NotificationManager.class);
        if (manager != null) {
            manager.createNotificationChannel(channel);
        }
    }

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        String action = intent == null ? null : intent.getAction();
        if (ACTION_STOP.equals(action)) {
            stopBot();
            return START_NOT_STICKY;
        }
        if (!ACTION_START.equals(action)) {
            stopSelf();
            return START_NOT_STICKY;
        }
        if (player != null) {
            return START_NOT_STICKY;
        }
        int resultCode = intent.getIntExtra(EXTRA_RESULT_CODE, 0);
        Intent resultData = parcelable(intent, EXTRA_DATA);
        startInForeground("Starting", "Preparing screen capture");
        worker = new HandlerThread("arrowbot", Process.THREAD_PRIORITY_DISPLAY);
        worker.start();
        workerHandler = new Handler(worker.getLooper());
        workerHandler.post(() -> startBot(resultCode, resultData));
        return START_NOT_STICKY;
    }

    @Override
    public IBinder onBind(Intent intent) {
        return null;
    }

    private void startInForeground(String title, String text) {
        Notification notification = buildNotification(title, text);
        if (Build.VERSION.SDK_INT >= 29) {
            startForeground(NOTIFICATION_ID, notification,
                    ServiceInfo.FOREGROUND_SERVICE_TYPE_MEDIA_PROJECTION);
        } else {
            startForeground(NOTIFICATION_ID, notification);
        }
        foregroundStarted = true;
        BotStatus.running = true;
        BotStatus.phase = title;
        BotStatus.detail = text;
    }

    private void startBot(int resultCode, Intent resultData) {
        try {
            if (stopping) {
                return;
            }
            if (resultCode == 0 || resultData == null) {
                fail("Screen capture was not allowed");
                return;
            }
            if (!waitForAccessibility()) {
                fail("Enable the Arrow AutoBot accessibility service first");
                return;
            }
            MediaProjectionManager manager =
                    (MediaProjectionManager) getSystemService(Context.MEDIA_PROJECTION_SERVICE);
            projection = manager.getMediaProjection(resultCode, resultData);
            if (projection == null) {
                fail("Could not get screen capture");
                return;
            }
            projection.registerCallback(new MediaProjection.Callback() {
                @Override
                public void onStop() {
                    mainHandler.post(() -> {
                        if (!stopping) {
                            fail("Screen capture was stopped");
                        }
                    });
                }
            }, workerHandler);

            capture = new ScreenCapture(projection, workerHandler);
            if (!capture.start(this)) {
                fail("Could not start screen capture");
                return;
            }
            BotStatus.captureWidth = capture.width();
            BotStatus.captureHeight = capture.height();
            BotStatus.failure = "";
            Config config = new Config().validate();
            player = new Player(config, new AndroidTransport(this, capture), listener);
            BotStatus.running = true;
            player.run();
            BotStatus.running = false;
            if (!stopping) {
                stopping = true;
                mainHandler.postDelayed(this::stopSelf, 400);
            }
        } catch (Throwable exc) {
            fail("Bot error: " + exc.getClass().getSimpleName()
                    + (exc.getMessage() == null ? "" : " - " + exc.getMessage()));
        }
    }

    private boolean waitForAccessibility() {
        for (int i = 0; i < 50; i++) {
            if (stopping) {
                return false;
            }
            if (BotAccessibilityService.getInstance() != null) {
                return true;
            }
            try {
                Thread.sleep(100);
            } catch (InterruptedException exc) {
                Thread.currentThread().interrupt();
                return false;
            }
        }
        return false;
    }

    private void fail(String message) {
        stopping = true;
        if (player != null) {
            player.stop();
        }
        if (capture != null) {
            capture.stop();
        }
        if (projection != null) {
            try {
                projection.stop();
            } catch (Exception ignored) {
            }
            projection = null;
        }
        BotStatus.running = false;
        BotStatus.phase = "Stopped";
        BotStatus.detail = message;
        BotStatus.failure = message;
        updateNotification("Stopped", message, true);
        mainHandler.postDelayed(this::stopSelf, 1800);
    }

    private void stopBot() {
        if (stopping) {
            return;
        }
        stopping = true;
        BotStatus.running = false;
        BotStatus.phase = "Stopped";
        BotStatus.detail = "";
        if (player != null) {
            player.stop();
        }
        if (capture != null) {
            capture.stop();
        }
        if (projection != null) {
            try {
                projection.stop();
            } catch (Exception ignored) {
            }
            projection = null;
        }
        NotificationManager manager = getSystemService(NotificationManager.class);
        if (manager != null) {
            manager.cancel(NOTIFICATION_ID);
        }
        stopForeground(true);
        mainHandler.postDelayed(this::stopSelf, 600);
    }

    @Override
    public void onDestroy() {
        stopping = true;
        if (player != null) {
            player.stop();
        }
        if (capture != null) {
            capture.stop();
        }
        if (projection != null) {
            try {
                projection.stop();
            } catch (Exception ignored) {
            }
            projection = null;
        }
        if (worker != null) {
            worker.quitSafely();
            worker = null;
        }
        BotStatus.running = false;
        if (foregroundStarted) {
            stopForeground(true);
        }
        super.onDestroy();
    }

    // --------------------------------------------------------------- listener
    private final BotListener listener = new BotListener() {
        @Override
        public void onPhase(String phase, String detail) {
            BotStatus.phase = phase;
            BotStatus.detail = detail;
            if (player != null) {
                BotStatus.levels = player.levelsCleared();
                BotStatus.taps = player.totalTaps();
                BotStatus.batches = player.totalBatches();
                BotStatus.lastSeconds = player.lastLevelSeconds();
                BotStatus.lastArrows = player.lastLevelArrows();
            }
            updateNotification(phase, detail, false);
        }

        @Override
        public void onBoard(Detection detection, int taps) {
            BotStatus.detail = detection.describe();
            updateNotification("Playing level",
                    detection.arrows.size() + " arrows ready", false);
        }

        @Override
        public void onLevelCleared(double seconds, int arrows, int taps) {
            BotStatus.levels++;
            BotStatus.lastSeconds = seconds;
            BotStatus.lastArrows = arrows;
            updateNotification("Level cleared", String.format(java.util.Locale.US,
                    "%.1f s · %d arrows · %d taps", seconds, arrows, taps), true);
        }

        @Override
        public void onFailure(String reason) {
            BotStatus.failure = reason;
        }
    };

    // ------------------------------------------------------------- notification
    private Notification buildNotification(String title, String text) {
        Intent open = new Intent(this, MainActivity.class)
                .addFlags(Intent.FLAG_ACTIVITY_SINGLE_TOP);
        PendingIntent openIntent = PendingIntent.getActivity(this, 0, open,
                PendingIntent.FLAG_UPDATE_CURRENT | PendingIntent.FLAG_IMMUTABLE);
        PendingIntent stopIntent = PendingIntent.getService(this, 1,
                new Intent(this, BotService.class).setAction(ACTION_STOP),
                PendingIntent.FLAG_UPDATE_CURRENT | PendingIntent.FLAG_IMMUTABLE);
        Notification.Action stopAction = new Notification.Action.Builder(
                Icon.createWithResource(this, R.drawable.ic_stat_arrow),
                getString(R.string.notification_stop), stopIntent).build();

        Notification.Builder builder = new Notification.Builder(this, CHANNEL_ID)
                .setSmallIcon(R.drawable.ic_stat_arrow)
                .setContentTitle(title)
                .setContentText(text)
                .setContentIntent(openIntent)
                .setCategory(Notification.CATEGORY_SERVICE)
                .setOngoing(true)
                .setOnlyAlertOnce(true)
                .setShowWhen(false)
                .setColor(getColor(R.color.accent))
                .addAction(stopAction);
        if (Build.VERSION.SDK_INT >= 31) {
            builder.setForegroundServiceBehavior(Notification.FOREGROUND_SERVICE_IMMEDIATE);
        }
        return builder.build();
    }

    private void updateNotification(String title, String text, boolean force) {
        if (!foregroundStarted) {
            return;
        }
        long now = SystemClock.elapsedRealtime();
        if (!force && now - lastNotificationAt < 800) {
            return;
        }
        lastNotificationAt = now;
        lastNotificationText = text;
        NotificationManager manager = getSystemService(NotificationManager.class);
        if (manager != null) {
            manager.notify(NOTIFICATION_ID, buildNotification(title, text));
        }
    }

    @SuppressWarnings("deprecation")
    private static Intent parcelable(Intent intent, String key) {
        if (Build.VERSION.SDK_INT >= 33) {
            return intent.getParcelableExtra(key, Intent.class);
        }
        return intent.getParcelableExtra(key);
    }
}
