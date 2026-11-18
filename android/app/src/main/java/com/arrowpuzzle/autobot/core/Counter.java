package com.arrowpuzzle.autobot.core;

import java.util.ArrayList;
import java.util.List;

/**
 * Read the remaining-arrow counter from the game's HUD.
 *
 * <p>The HUD font is fixed, so ten 20x32 glyph templates are enough.  A low
 * score means "unreadable", never "guess": the agent treats an unreadable
 * counter as agreement, so a HUD animation can never dead-end a run.</p>
 */
public final class Counter {
    public static final int GLYPH_W = 20;
    public static final int GLYPH_H = 32;
    public static final int GRAY_MAX = 150;

    private static final boolean[][] TEMPLATES = buildTemplates();

    private Counter() {
    }

    /** Return the counter value, or {@code null} when it cannot be read surely. */
    public static Integer read(Frame frame, Config config) {
        if (frame == null || !config.counterCheck) {
            return null;
        }
        int[] box = Detector.fractionsToBox(config.counterRoi, frame.width, frame.height);
        int left = box[0];
        int top = box[1];
        int width = Math.max(1, box[2] - box[0]);
        int height = Math.max(1, box[3] - box[1]);
        byte[] mask = new byte[width * height];
        int index = 0;
        for (int y = 0; y < height; y++) {
            int row = (top + y) * frame.width + left;
            for (int x = 0; x < width; x++) {
                int pixel = frame.pixels[row + x];
                int r = (pixel >>> 16) & 0xff;
                int g = (pixel >>> 8) & 0xff;
                int b = pixel & 0xff;
                mask[index++] = (byte) (299 * r + 587 * g + 114 * b < GRAY_MAX * 1000 ? 1 : 0);
            }
        }

        Components components = new Components();
        components.label(mask, width, height, true);
        List<Glyph> glyphs = new ArrayList<>();
        for (int component = 1; component <= components.count; component++) {
            int x = components.x[component];
            int y = components.y[component];
            int w = components.w[component];
            int h = components.h[component];
            int area = components.area[component];
            if (area < 15 || h < 12 || w < 4) {
                continue;
            }
            if (w / (float) h > 0.85f) {
                continue; // the wide arrow icon is not a digit
            }
            float[] patch = new float[w * h];
            for (int py = 0; py < h; py++) {
                int row = (y + py) * width + x;
                for (int px = 0; px < w; px++) {
                    if (components.labels[row + px] == component) {
                        patch[py * w + px] = 1.0f;
                    }
                }
            }
            glyphs.add(new Glyph(x, resize(patch, w, h)));
        }
        if (glyphs.isEmpty() || glyphs.size() > 3) {
            return null;
        }
        glyphs.sort((a, b) -> Integer.compare(a.left, b.left));
        StringBuilder text = new StringBuilder();
        for (Glyph glyph : glyphs) {
            int bestDigit = -1;
            float bestScore = -1f;
            for (int digit = 0; digit < TEMPLATES.length; digit++) {
                float score = match(glyph.pixels, TEMPLATES[digit]);
                if (score > bestScore) {
                    bestScore = score;
                    bestDigit = digit;
                }
            }
            if (bestScore < config.counterMinScore) {
                return null;
            }
            text.append((char) ('0' + bestDigit));
        }
        try {
            return Integer.parseInt(text.toString());
        } catch (NumberFormatException exc) {
            return null;
        }
    }

    private static float match(float[] glyph, boolean[] template) {
        int agree = 0;
        for (int i = 0; i < glyph.length; i++) {
            if ((glyph[i] > 0.5f) == template[i]) {
                agree++;
            }
        }
        return agree / (float) glyph.length;
    }

    private static float[] resize(float[] source, int sw, int sh) {
        // OpenCV's INTER_AREA falls back to bilinear when both axes enlarge.
        if (sw >= GLYPH_W && sh >= GLYPH_H) {
            return resizeArea(source, sw, sh);
        }
        return resizeBilinear(source, sw, sh);
    }

    private static float[] resizeArea(float[] source, int sw, int sh) {
        float[] out = new float[GLYPH_W * GLYPH_H];
        float scaleX = sw / (float) GLYPH_W;
        float scaleY = sh / (float) GLYPH_H;
        for (int dy = 0; dy < GLYPH_H; dy++) {
            double y0 = dy * scaleY;
            double y1 = Math.min(sh, (dy + 1) * scaleY);
            int sy0 = (int) Math.floor(y0);
            int sy1 = (int) Math.ceil(y1);
            for (int dx = 0; dx < GLYPH_W; dx++) {
                double x0 = dx * scaleX;
                double x1 = Math.min(sw, (dx + 1) * scaleX);
                int sx0 = (int) Math.floor(x0);
                int sx1 = (int) Math.ceil(x1);
                double sum = 0.0;
                double weight = 0.0;
                for (int sy = sy0; sy < sy1; sy++) {
                    double wy = Math.min(y1, sy + 1) - Math.max(y0, sy);
                    if (wy <= 0) {
                        continue;
                    }
                    for (int sx = sx0; sx < sx1; sx++) {
                        double wx = Math.min(x1, sx + 1) - Math.max(x0, sx);
                        if (wx <= 0) {
                            continue;
                        }
                        double w = wx * wy;
                        sum += source[sy * sw + sx] * w;
                        weight += w;
                    }
                }
                out[dy * GLYPH_W + dx] = weight > 0 ? (float) (sum / weight) : 0f;
            }
        }
        return out;
    }

    private static float[] resizeBilinear(float[] source, int sw, int sh) {
        float[] out = new float[GLYPH_W * GLYPH_H];
        for (int dy = 0; dy < GLYPH_H; dy++) {
            float sy = (dy + 0.5f) * sh / GLYPH_H - 0.5f;
            sy = Config.clamp(sy, 0f, sh - 1f);
            int y0 = (int) Math.floor(sy);
            int y1 = Math.min(sh - 1, y0 + 1);
            float fy = sy - y0;
            for (int dx = 0; dx < GLYPH_W; dx++) {
                float sx = (dx + 0.5f) * sw / GLYPH_W - 0.5f;
                sx = Config.clamp(sx, 0f, sw - 1f);
                int x0 = (int) Math.floor(sx);
                int x1 = Math.min(sw - 1, x0 + 1);
                float fx = sx - x0;
                float top = source[y0 * sw + x0] * (1f - fx) + source[y0 * sw + x1] * fx;
                float bottom = source[y1 * sw + x0] * (1f - fx) + source[y1 * sw + x1] * fx;
                out[dy * GLYPH_W + dx] = top * (1f - fy) + bottom * fy;
            }
        }
        return out;
    }

    private static boolean[][] buildTemplates() {
        boolean[][] templates = new boolean[10][GLYPH_W * GLYPH_H];
        for (int digit = 0; digit < 10; digit++) {
            String[] rows = DigitTemplates.ART[digit];
            if (rows.length != GLYPH_H) {
                throw new IllegalStateException("template " + digit + " has wrong height");
            }
            for (int y = 0; y < GLYPH_H; y++) {
                if (rows[y].length() != GLYPH_W) {
                    throw new IllegalStateException("template " + digit + " has wrong width");
                }
                for (int x = 0; x < GLYPH_W; x++) {
                    templates[digit][y * GLYPH_W + x] = rows[y].charAt(x) == '#';
                }
            }
        }
        return templates;
    }

    private static final class Glyph {
        final int left;
        final float[] pixels;

        Glyph(int left, float[] pixels) {
            this.left = left;
            this.pixels = pixels;
        }
    }
}
