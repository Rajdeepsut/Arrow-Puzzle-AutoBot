package com.arrowpuzzle.autobot;

import android.content.Context;
import android.graphics.PixelFormat;
import android.graphics.Rect;
import android.hardware.display.DisplayManager;
import android.hardware.display.VirtualDisplay;
import android.media.Image;
import android.media.ImageReader;
import android.media.projection.MediaProjection;
import android.os.Build;
import android.os.Handler;
import android.os.SystemClock;
import android.util.DisplayMetrics;
import android.view.WindowManager;

import com.arrowpuzzle.autobot.core.Frame;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.IntBuffer;

/**
 * MediaProjection screen capture.  Keeps one ImageReader and one reusable
 * {@link Frame}; RGBA_8888 pixels are converted to 0xAARRGGBB once per frame.
 */
final class ScreenCapture {
    private final MediaProjection projection;
    private final Handler handler;
    private final Frame frame = new Frame(1, 1);
    private ImageReader reader;
    private VirtualDisplay display;
    private int width;
    private int height;

    ScreenCapture(MediaProjection projection, Handler handler) {
        this.projection = projection;
        this.handler = handler;
    }

    boolean start(Context context) {
        WindowManager manager = (WindowManager) context.getSystemService(Context.WINDOW_SERVICE);
        if (Build.VERSION.SDK_INT >= 30) {
            Rect bounds = manager.getMaximumWindowMetrics().getBounds();
            width = bounds.width();
            height = bounds.height();
        } else {
            DisplayMetrics metrics = new DisplayMetrics();
            manager.getDefaultDisplay().getRealMetrics(metrics);
            width = metrics.widthPixels;
            height = metrics.heightPixels;
        }
        if (width <= 0 || height <= 0) {
            return false;
        }
        int density = context.getResources().getDisplayMetrics().densityDpi;
        try {
            reader = ImageReader.newInstance(width, height, PixelFormat.RGBA_8888, 3);
            display = projection.createVirtualDisplay("ArrowBot", width, height, density,
                    DisplayManager.VIRTUAL_DISPLAY_FLAG_AUTO_MIRROR,
                    reader.getSurface(), null, handler);
            return display != null;
        } catch (Exception exc) {
            stop();
            return false;
        }
    }

    int width() {
        return width;
    }

    int height() {
        return height;
    }

    /** Latest frame, or null if the reader produced nothing before the deadline. */
    Frame capture(int timeoutMs) {
        if (reader == null) {
            return null;
        }
        long deadline = SystemClock.elapsedRealtime() + Math.max(100, timeoutMs);
        while (SystemClock.elapsedRealtime() < deadline) {
            Image image = null;
            try {
                image = reader.acquireLatestImage();
            } catch (Exception ignored) {
                return null;
            }
            if (image != null) {
                try {
                    copy(image);
                    return frame;
                } finally {
                    image.close();
                }
            }
            try {
                Thread.sleep(12);
            } catch (InterruptedException exc) {
                Thread.currentThread().interrupt();
                return null;
            }
        }
        return null;
    }

    private void copy(Image image) {
        Image.Plane plane = image.getPlanes()[0];
        ByteBuffer buffer = plane.getBuffer();
        int pixelStride = plane.getPixelStride();
        int rowStride = plane.getRowStride();
        int imageWidth = image.getWidth();
        int imageHeight = image.getHeight();
        frame.ensure(imageWidth, imageHeight);
        int[] out = frame.pixels;

        if (pixelStride == 4) {
            buffer.rewind();
            IntBuffer ints = buffer.order(ByteOrder.LITTLE_ENDIAN).asIntBuffer();
            int rowInts = rowStride / 4;
            for (int y = 0; y < imageHeight; y++) {
                ints.position(y * rowInts);
                ints.get(out, y * imageWidth, imageWidth);
            }
            for (int i = 0; i < out.length; i++) {
                int value = out[i];
                int r = value & 0xff;
                int g = (value >>> 8) & 0xff;
                int b = (value >>> 16) & 0xff;
                out[i] = 0xff000000 | (r << 16) | (g << 8) | b;
            }
        } else {
            for (int y = 0; y < imageHeight; y++) {
                int row = y * rowStride;
                for (int x = 0; x < imageWidth; x++) {
                    int offset = row + x * pixelStride;
                    int r = buffer.get(offset) & 0xff;
                    int g = buffer.get(offset + 1) & 0xff;
                    int b = buffer.get(offset + 2) & 0xff;
                    out[y * imageWidth + x] = 0xff000000 | (r << 16) | (g << 8) | b;
                }
            }
        }
    }

    void stop() {
        if (display != null) {
            try {
                display.release();
            } catch (Exception ignored) {
            }
            display = null;
        }
        if (reader != null) {
            try {
                reader.close();
            } catch (Exception ignored) {
            }
            reader = null;
        }
    }
}
