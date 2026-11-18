package com.arrowpuzzle.autobot.core;

/** One screen capture, in 0xAARRGGBB pixels.  Reused between frames. */
public final class Frame {
    public int width;
    public int height;
    public int[] pixels;

    public Frame(int width, int height) {
        ensure(width, height);
    }

    public void ensure(int width, int height) {
        if (width <= 0 || height <= 0) {
            throw new IllegalArgumentException("bad frame size " + width + "x" + height);
        }
        this.width = width;
        this.height = height;
        int needed = width * height;
        if (pixels == null || pixels.length != needed) {
            pixels = new int[needed];
        }
    }

    public int red(int x, int y) {
        return (pixels[y * width + x] >>> 16) & 0xff;
    }

    public int green(int x, int y) {
        return (pixels[y * width + x] >>> 8) & 0xff;
    }

    public int blue(int x, int y) {
        return pixels[y * width + x] & 0xff;
    }
}
