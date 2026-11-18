package com.arrowpuzzle.autobot.core;

import java.util.Arrays;

/**
 * 8-connected binary component labelling with a reusable flood-fill queue.
 * Used both by the board detector (full board ROI) and the HUD counter (tiny).
 */
final class Components {
    int width;
    int height;
    int count;
    int[] labels;
    int[] x;
    int[] y;
    int[] w;
    int[] h;
    int[] area;
    private int[] queue;
    private int capacity = 64;

    Components() {
        allocateStats(capacity);
    }

    private void allocateStats(int size) {
        x = new int[size];
        y = new int[size];
        w = new int[size];
        h = new int[size];
        area = new int[size];
    }

    private void ensureCapacity(int needed) {
        if (needed <= capacity) {
            return;
        }
        int next = capacity;
        while (next < needed) {
            next *= 2;
        }
        int[] ox = x;
        int[] oy = y;
        int[] ow = w;
        int[] oh = h;
        int[] oa = area;
        allocateStats(next);
        System.arraycopy(ox, 0, x, 0, capacity);
        System.arraycopy(oy, 0, y, 0, capacity);
        System.arraycopy(ow, 0, w, 0, capacity);
        System.arraycopy(oh, 0, h, 0, capacity);
        System.arraycopy(oa, 0, area, 0, capacity);
        capacity = next;
    }

    /** Labels every non-zero pixel; the caller reads {@link #count} and the stats. */
    void label(byte[] mask, int width, int height, boolean eight) {
        this.width = width;
        this.height = height;
        int total = width * height;
        if (labels == null || labels.length != total) {
            labels = new int[total];
        } else {
            Arrays.fill(labels, 0);
        }
        if (queue == null || queue.length < total) {
            queue = new int[total];
        }
        count = 0;
        for (int start = 0; start < total; start++) {
            if (mask[start] == 0 || labels[start] != 0) {
                continue;
            }
            count++;
            ensureCapacity(count + 1);
            int head = 0;
            int tail = 0;
            queue[tail++] = start;
            labels[start] = count;
            int minX = start % width;
            int maxX = minX;
            int startY = start / width;
            int minY = startY;
            int maxY = startY;
            int pixels = 0;

            while (head < tail) {
                int index = queue[head++];
                int cursorY = index / width;
                int cursorX = index - cursorY * width;
                pixels++;
                if (cursorX < minX) minX = cursorX;
                if (cursorX > maxX) maxX = cursorX;
                if (cursorY < minY) minY = cursorY;
                if (cursorY > maxY) maxY = cursorY;

                if (cursorX > 0 && visit(mask, labels, index - 1, count, tail)) {
                    queue[tail++] = index - 1;
                }
                if (cursorX + 1 < width && visit(mask, labels, index + 1, count, tail)) {
                    queue[tail++] = index + 1;
                }
                if (cursorY > 0 && visit(mask, labels, index - width, count, tail)) {
                    queue[tail++] = index - width;
                }
                if (cursorY + 1 < height && visit(mask, labels, index + width, count, tail)) {
                    queue[tail++] = index + width;
                }
                if (eight) {
                    if (cursorX > 0 && cursorY > 0
                            && visit(mask, labels, index - width - 1, count, tail)) {
                        queue[tail++] = index - width - 1;
                    }
                    if (cursorX + 1 < width && cursorY > 0
                            && visit(mask, labels, index - width + 1, count, tail)) {
                        queue[tail++] = index - width + 1;
                    }
                    if (cursorX > 0 && cursorY + 1 < height
                            && visit(mask, labels, index + width - 1, count, tail)) {
                        queue[tail++] = index + width - 1;
                    }
                    if (cursorX + 1 < width && cursorY + 1 < height
                            && visit(mask, labels, index + width + 1, count, tail)) {
                        queue[tail++] = index + width + 1;
                    }
                }
            }
            x[count] = minX;
            y[count] = minY;
            w[count] = maxX - minX + 1;
            h[count] = maxY - minY + 1;
            area[count] = pixels;
        }
    }

    private static boolean visit(byte[] mask, int[] labels, int index, int label, int ignored) {
        if (mask[index] == 0 || labels[index] != 0) {
            return false;
        }
        labels[index] = label;
        return true;
    }
}
