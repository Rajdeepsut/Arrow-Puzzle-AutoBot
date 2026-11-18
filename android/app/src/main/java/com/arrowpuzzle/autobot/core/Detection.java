package com.arrowpuzzle.autobot.core;

import java.util.List;

/** Everything one screenshot said about the board. */
public final class Detection {
    /** The capture this reading came from; it is reused by the next capture. */
    public final Frame frame;
    public final float scale;
    public final List<Arrow> arrows;
    /** Inclusive ink bounding box in absolute screen pixels, or all zero. */
    public final int x0;
    public final int y0;
    public final int x1;
    public final int y1;

    public Detection(Frame frame, float scale, List<Arrow> arrows,
                     int x0, int y0, int x1, int y1) {
        this.frame = frame;
        this.scale = scale;
        this.arrows = arrows;
        this.x0 = x0;
        this.y0 = y0;
        this.x1 = x1;
        this.y1 = y1;
    }

    /** True when both frames show the same, unmoved board. */
    public boolean sameView(Detection other, int tolerance) {
        if (other == null || arrows.size() != other.arrows.size()) {
            return false;
        }
        if (Math.abs(x0 - other.x0) > tolerance || Math.abs(y0 - other.y0) > tolerance
                || Math.abs(x1 - other.x1) > tolerance
                || Math.abs(y1 - other.y1) > tolerance) {
            return false;
        }
        int[][] mine = sortedHeads(arrows);
        int[][] theirs = sortedHeads(other.arrows);
        for (int i = 0; i < mine.length; i++) {
            if (Math.abs(mine[i][0] - theirs[i][0]) > tolerance
                    || Math.abs(mine[i][1] - theirs[i][1]) > tolerance) {
                return false;
            }
        }
        return true;
    }

    static int[][] sortedHeads(List<Arrow> arrows) {
        int[][] heads = new int[arrows.size()][2];
        for (int i = 0; i < arrows.size(); i++) {
            heads[i][0] = arrows.get(i).headX;
            heads[i][1] = arrows.get(i).headY;
        }
        java.util.Arrays.sort(heads, (a, b) -> a[0] != b[0]
                ? Integer.compare(a[0], b[0]) : Integer.compare(a[1], b[1]));
        return heads;
    }

    public String describe() {
        int up = 0;
        int right = 0;
        int down = 0;
        int left = 0;
        int low = 0;
        for (Arrow arrow : arrows) {
            switch (arrow.direction) {
                case 'U': up++; break;
                case 'R': right++; break;
                case 'D': down++; break;
                default: left++; break;
            }
            if (arrow.confidence < 0.10f) {
                low++;
            }
        }
        String text = arrows.size() + " arrows | scale=" + String.format(java.util.Locale.US, "%.2f", scale)
                + " | U" + up + " R" + right + " D" + down + " L" + left
                + " | bbox=(" + x0 + "," + y0 + "," + x1 + "," + y1 + ")";
        return low == 0 ? text : text + " | " + low + " uncertain";
    }
}
