package com.arrowpuzzle.autobot.core;

/** One arrow: where its head is, where it points, and the ink it owns. */
public final class Arrow {
    public static final char[] DIRECTIONS = {'U', 'R', 'D', 'L'};

    public final int headX;
    public final int headY;
    public char direction;
    /** Template margin over the runner-up; used as the confidence gate. */
    public final float confidence;
    public final float scale;
    /** Absolute screen coordinates of the ink this arrow owns. */
    public final int[] xs;
    public final int[] ys;
    /** Template score for every direction, indexed as {@link #DIRECTIONS}. */
    public final float[] scores;

    public Arrow(int headX, int headY, char direction, float confidence, float scale,
                 int[] xs, int[] ys, float[] scores) {
        this.headX = headX;
        this.headY = headY;
        this.direction = direction;
        this.confidence = confidence;
        this.scale = scale;
        this.xs = xs;
        this.ys = ys;
        this.scores = scores;
    }

    public int size() {
        return xs.length;
    }

    public Arrow withDirection(char newDirection) {
        return new Arrow(headX, headY, newDirection, confidence, scale, xs, ys, scores);
    }

    public static int indexOf(char direction) {
        for (int i = 0; i < DIRECTIONS.length; i++) {
            if (DIRECTIONS[i] == direction) {
                return i;
            }
        }
        return 0;
    }

    @Override
    public String toString() {
        return direction + "@(" + headX + "," + headY + ")";
    }
}
