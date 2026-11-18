package com.arrowpuzzle.autobot.core;

/** Status callbacks for the notification and the in-app screen. */
public interface BotListener {
    void onPhase(String phase, String detail);

    void onBoard(Detection detection, int taps);

    void onLevelCleared(double seconds, int arrows, int taps);

    void onFailure(String reason);
}
