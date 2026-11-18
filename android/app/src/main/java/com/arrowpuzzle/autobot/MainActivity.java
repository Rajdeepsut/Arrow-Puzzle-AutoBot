package com.arrowpuzzle.autobot;

import android.Manifest;
import android.app.Activity;
import android.app.AlertDialog;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.media.projection.MediaProjectionConfig;
import android.media.projection.MediaProjectionManager;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.os.PowerManager;
import android.provider.Settings;
import android.view.View;
import android.widget.Button;
import android.widget.CheckBox;
import android.widget.TextView;
import android.widget.Toast;

import java.util.Locale;

/** Setup screen: permissions, start/stop, and live status. */
public class MainActivity extends Activity {
    private static final int REQUEST_CAPTURE = 100;
    private static final int REQUEST_NOTIFICATIONS = 101;

    private final Handler handler = new Handler(Looper.getMainLooper());
    private TextView tvPhase;
    private TextView tvDetail;
    private TextView tvStats;
    private TextView tvAccessibility;
    private TextView tvCapture;
    private TextView tvBattery;
    private TextView tvHint;
    private Button btnStart;
    private Button btnStop;
    private CheckBox cbOpenGame;
    private boolean pendingStart;

    private final Runnable refresh = new Runnable() {
        @Override
        public void run() {
            updateUi();
            handler.postDelayed(this, 500);
        }
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        tvPhase = findViewById(R.id.tvPhase);
        tvDetail = findViewById(R.id.tvDetail);
        tvStats = findViewById(R.id.tvStats);
        tvAccessibility = findViewById(R.id.tvAccessibility);
        tvCapture = findViewById(R.id.tvCapture);
        tvBattery = findViewById(R.id.tvBattery);
        tvHint = findViewById(R.id.tvHint);
        btnStart = findViewById(R.id.btnStart);
        btnStop = findViewById(R.id.btnStop);
        cbOpenGame = findViewById(R.id.cbOpenGame);

        btnStart.setOnClickListener(view -> beginStart());
        btnStop.setOnClickListener(view -> stopBot());
        findViewById(R.id.btnAccessibility).setOnClickListener(view -> openAccessibility());
        findViewById(R.id.btnBattery).setOnClickListener(view -> fixBattery());
        findViewById(R.id.btnGame).setOnClickListener(view -> openGame());
        updateUi();
    }

    @Override
    protected void onResume() {
        super.onResume();
        handler.removeCallbacks(refresh);
        handler.post(refresh);
    }

    @Override
    protected void onPause() {
        super.onPause();
        handler.removeCallbacks(refresh);
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        handler.removeCallbacksAndMessages(null);
    }

    // ------------------------------------------------------------------ start
    private void beginStart() {
        if (BotStatus.running) {
            toast(getString(R.string.already_running));
            return;
        }
        if (!BotAccessibilityService.isEnabled(this)) {
            showAccessibilityDialog();
            return;
        }
        if (Build.VERSION.SDK_INT >= 33
                && checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS)
                != PackageManager.PERMISSION_GRANTED) {
            pendingStart = true;
            requestPermissions(new String[]{Manifest.permission.POST_NOTIFICATIONS},
                    REQUEST_NOTIFICATIONS);
            return;
        }
        requestCapture();
    }

    private void requestCapture() {
        MediaProjectionManager manager =
                (MediaProjectionManager) getSystemService(MEDIA_PROJECTION_SERVICE);
        Intent intent;
        if (Build.VERSION.SDK_INT >= 34) {
            intent = manager.createScreenCaptureIntent(
                    MediaProjectionConfig.createConfigForDefaultDisplay());
        } else {
            intent = manager.createScreenCaptureIntent();
        }
        try {
            startActivityForResult(intent, REQUEST_CAPTURE);
        } catch (Exception exc) {
            toast("Screen capture is unavailable on this device");
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions,
                                           int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == REQUEST_NOTIFICATIONS) {
            pendingStart = false;
            if (grantResults.length > 0
                    && grantResults[0] != PackageManager.PERMISSION_GRANTED) {
                toast(getString(R.string.notification_permission));
            }
            requestCapture();
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode != REQUEST_CAPTURE) {
            return;
        }
        if (resultCode != RESULT_OK || data == null) {
            toast(getString(R.string.capture_cancelled));
            return;
        }
        Intent service = new Intent(this, BotService.class)
                .setAction(BotService.ACTION_START)
                .putExtra(BotService.EXTRA_RESULT_CODE, resultCode)
                .putExtra(BotService.EXTRA_DATA, data);
        try {
            startForegroundService(service);
            BotStatus.running = true;
            BotStatus.phase = "Starting";
            BotStatus.detail = "Waiting for Arrow Puzzle";
            BotStatus.failure = "";
            updateUi();
            if (cbOpenGame.isChecked()) {
                handler.postDelayed(this::openGame, 1600);
            }
        } catch (Throwable exc) {
            toast("Could not start the bot: " + exc.getMessage());
            BotStatus.running = false;
        }
    }

    private void stopBot() {
        if (BotStatus.running) {
            try {
                startService(new Intent(this, BotService.class)
                        .setAction(BotService.ACTION_STOP));
            } catch (Exception ignored) {
                stopService(new Intent(this, BotService.class));
            }
        } else {
            stopService(new Intent(this, BotService.class));
        }
        BotStatus.running = false;
        BotStatus.phase = "Stopped";
        BotStatus.detail = "";
        updateUi();
    }

    // ------------------------------------------------------------ permissions
    private void showAccessibilityDialog() {
        new AlertDialog.Builder(this)
                .setTitle(R.string.accessibility_dialog_title)
                .setMessage(R.string.accessibility_dialog_message)
                .setPositiveButton(R.string.open_settings,
                        (dialog, which) -> openAccessibility())
                .setNegativeButton(R.string.cancel, null)
                .show();
    }

    private void openAccessibility() {
        try {
            startActivity(new Intent(Settings.ACTION_ACCESSIBILITY_SETTINGS));
        } catch (Exception exc) {
            toast("Open Settings → Accessibility → Installed services");
        }
    }

    private void fixBattery() {
        PowerManager power = (PowerManager) getSystemService(POWER_SERVICE);
        if (power != null && power.isIgnoringBatteryOptimizations(getPackageName())) {
            toast("Battery use is already unrestricted");
            return;
        }
        try {
            startActivity(new Intent(
                    Settings.ACTION_REQUEST_IGNORE_BATTERY_OPTIMIZATIONS,
                    Uri.parse("package:" + getPackageName())));
        } catch (Exception exc) {
            try {
                startActivity(new Intent(
                        Settings.ACTION_IGNORE_BATTERY_OPTIMIZATION_SETTINGS));
            } catch (Exception ignored) {
                toast("Open Settings → Battery → Unrestricted");
            }
        }
    }

    private void openGame() {
        Intent launch = getPackageManager().getLaunchIntentForPackage(AppConfig.GAME_PACKAGE);
        if (launch == null) {
            toast(getString(R.string.game_missing));
            return;
        }
        launch.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK
                | Intent.FLAG_ACTIVITY_RESET_TASK_IF_NEEDED);
        try {
            startActivity(launch);
        } catch (Exception exc) {
            toast(getString(R.string.game_missing));
        }
    }

    // ---------------------------------------------------------------------- ui
    private void updateUi() {
        boolean running = BotStatus.running;
        boolean accessibility = BotAccessibilityService.isEnabled(this);
        PowerManager power = (PowerManager) getSystemService(POWER_SERVICE);
        boolean batteryFree = power != null
                && power.isIgnoringBatteryOptimizations(getPackageName());

        String phase;
        if (running) {
            phase = BotStatus.phase == null || BotStatus.phase.isEmpty()
                    ? "Running" : BotStatus.phase;
        } else if ("Stopped".equals(BotStatus.phase)) {
            phase = "Stopped";
        } else {
            phase = "Idle";
        }
        tvPhase.setText(phase);
        tvPhase.setTextColor(getColor(running ? R.color.accent : R.color.text));
        String detail = BotStatus.detail == null || BotStatus.detail.isEmpty()
                ? getString(R.string.start_hint) : BotStatus.detail;
        tvDetail.setText(detail);

        String last = BotStatus.lastSeconds > 0
                ? String.format(Locale.US, "%.1f s · %d arrows",
                BotStatus.lastSeconds, BotStatus.lastArrows)
                : "—";
        tvStats.setText(String.format(Locale.US,
                "Levels %d   Taps %d   Batches %d%nLast level %s",
                BotStatus.levels, BotStatus.taps, BotStatus.batches, last));

        tvAccessibility.setText(accessibility
                ? getString(R.string.enabled) : getString(R.string.disabled));
        tvAccessibility.setTextColor(getColor(
                accessibility ? R.color.accent : R.color.warning));

        if (running) {
            int width = BotStatus.captureWidth;
            int height = BotStatus.captureHeight;
            tvCapture.setText(width > 0
                    ? getString(R.string.active) + " · " + width + "×" + height
                    : getString(R.string.active));
            tvCapture.setTextColor(getColor(R.color.accent));
        } else {
            tvCapture.setText("Asked when you press Start");
            tvCapture.setTextColor(getColor(R.color.muted));
        }

        tvBattery.setText(batteryFree
                ? getString(R.string.unrestricted) : getString(R.string.optimized));
        tvBattery.setTextColor(getColor(
                batteryFree ? R.color.accent : R.color.warning));

        btnStart.setEnabled(!running);
        btnStart.setAlpha(running ? 0.45f : 1f);
        btnStop.setEnabled(running);
        btnStop.setAlpha(running ? 1f : 0.45f);

        if (!accessibility) {
            tvHint.setText("Accessibility is required. Enable “Arrow AutoBot gestures” "
                    + "in Settings → Accessibility → Installed services.");
            tvHint.setTextColor(getColor(R.color.warning));
        } else if (!batteryFree) {
            tvHint.setText("Tip: on vivo/OriginOS also allow auto-start and background "
                    + "activity, and set battery use to Unrestricted.");
            tvHint.setTextColor(getColor(R.color.warning));
        } else {
            tvHint.setText("Ready. Leave Arrow Puzzle in the foreground; the bot only "
                    + "taps while it can see a stable board.");
            tvHint.setTextColor(getColor(R.color.muted));
        }
    }

    private void toast(String message) {
        Toast.makeText(this, message, Toast.LENGTH_SHORT).show();
    }
}
