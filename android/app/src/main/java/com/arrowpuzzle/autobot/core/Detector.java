package com.arrowpuzzle.autobot.core;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Turns a screen capture into a {@link Detection}.
 *
 * <p>This is a direct port of the proven Python detector: ink mask, exact
 * Euclidean distance transform, one maximum per thick head, geodesic Voronoi
 * split of touching arrows, and a synthetic triangle match for the direction.
 * The board region is cropped first, so all work happens on ~1.7 M pixels.</p>
 */
public final class Detector {
    public static final float HEAD_DT = 5.0f;
    public static final float HEAD_APEX_OFFSET = 8.0f;
    public static final float HEAD_HEIGHT = 11.0f;
    public static final float HEAD_BASE = 15.0f;
    public static final float SHAFT_HALF_WIDTH = 1.5f;
    public static final float SHAFT_STUB = 10.0f;
    public static final float PATCH_RADIUS = 13.0f;

    private static final float FAR = 1.0e12f;

    private final Config config;

    private int roiX;
    private int roiY;
    private int roiW;
    private int roiH;
    private byte[] mask;
    private byte[] headMask;
    private float[] distSq;
    private float[] lineIn;
    private float[] lineOut;
    private double[] envelope;
    private int[] envelopeIndex;
    private final Components blobs = new Components();
    private final Components headBlobs = new Components();
    private int inkCount;
    private int inkX0;
    private int inkY0;
    private int inkX1;
    private int inkY1;

    public Detector(Config config) {
        this.config = config;
        config.validate();
    }

    public Detection detect(Frame frame) {
        int[] box = fractionsToBox(config.roi, frame.width, frame.height);
        roiX = box[0];
        roiY = box[1];
        roiW = Math.max(1, box[2] - box[0]);
        roiH = Math.max(1, box[3] - box[1]);
        ensureBuffers();

        buildMask(frame);
        edt();
        float maxSq = 0f;
        for (float value : distSq) {
            if (value > maxSq) {
                maxSq = value;
            }
        }
        float scale = maxSq > 0f ? (float) (Math.sqrt(maxSq) / HEAD_DT) : 1.0f;
        scale = Config.clamp(scale, config.scaleMin, config.scaleMax);
        List<Arrow> arrows = detectArrows(scale);
        if (inkCount == 0) {
            return new Detection(frame, scale, arrows, 0, 0, 0, 0);
        }
        return new Detection(frame, scale, arrows,
                inkX0 + roiX, inkY0 + roiY, inkX1 + roiX, inkY1 + roiY);
    }

    private void ensureBuffers() {
        int total = roiW * roiH;
        if (mask == null || mask.length != total) {
            mask = new byte[total];
            headMask = new byte[total];
            distSq = new float[total];
        }
        if (lineIn == null || lineIn.length < Math.max(roiW, roiH)) {
            int longest = Math.max(roiW, roiH);
            lineIn = new float[longest];
            lineOut = new float[longest];
            envelope = new double[longest + 1];
            envelopeIndex = new int[longest];
        }
    }

    private void buildMask(Frame frame) {
        int grayMax = config.inkGrayMax;
        int limit = grayMax * 1000;
        boolean redInk = config.redInk;
        inkCount = 0;
        inkX0 = roiW;
        inkY0 = roiH;
        inkX1 = -1;
        inkY1 = -1;
        int index = 0;
        for (int y = 0; y < roiH; y++) {
            int row = (roiY + y) * frame.width + roiX;
            for (int x = 0; x < roiW; x++) {
                int pixel = frame.pixels[row + x];
                int r = (pixel >>> 16) & 0xff;
                int g = (pixel >>> 8) & 0xff;
                int b = pixel & 0xff;
                boolean ink = 299 * r + 587 * g + 114 * b < limit;
                if (!ink && redInk
                        && r > 130 && g < 130 && b < 130 && r - g > 40 && r - b > 40) {
                    ink = true;
                }
                if (ink) {
                    mask[index] = 1;
                    inkCount++;
                    if (x < inkX0) inkX0 = x;
                    if (x > inkX1) inkX1 = x;
                    if (y < inkY0) inkY0 = y;
                    if (y > inkY1) inkY1 = y;
                } else {
                    mask[index] = 0;
                }
                index++;
            }
        }
    }

    /**
     * Exact squared Euclidean distance to the nearest background pixel
     * (Felzenszwalb-Huttenlocher).  OpenCV's precise transform is the reference.
     */
    private void edt() {
        for (int x = 0; x < roiW; x++) {
            for (int y = 0; y < roiH; y++) {
                lineIn[y] = mask[y * roiW + x] != 0 ? FAR : 0f;
            }
            transform1d(roiH);
            for (int y = 0; y < roiH; y++) {
                distSq[y * roiW + x] = lineOut[y];
            }
        }
        for (int y = 0; y < roiH; y++) {
            System.arraycopy(distSq, y * roiW, lineIn, 0, roiW);
            transform1d(roiW);
            System.arraycopy(lineOut, 0, distSq, y * roiW, roiW);
        }
    }

    private void transform1d(int length) {
        int k = 0;
        envelopeIndex[0] = 0;
        envelope[0] = -1.0e30;
        envelope[1] = 1.0e30;
        for (int q = 1; q < length; q++) {
            double fq = lineIn[q] + (double) q * q;
            double intersection;
            while (true) {
                int vk = envelopeIndex[k];
                intersection = (fq - (lineIn[vk] + (double) vk * vk))
                        / (2.0 * q - 2.0 * vk);
                if (intersection <= envelope[k] && k > 0) {
                    k--;
                } else {
                    break;
                }
            }
            k++;
            envelopeIndex[k] = q;
            envelope[k] = intersection;
            envelope[k + 1] = 1.0e30;
        }
        k = 0;
        for (int q = 0; q < length; q++) {
            while (envelope[k + 1] < q) {
                k++;
            }
            int dq = q - envelopeIndex[k];
            lineOut[q] = (float) ((double) dq * dq + lineIn[envelopeIndex[k]]);
        }
    }

    private List<Arrow> detectArrows(float scale) {
        int radius = Math.max(4, Math.round(PATCH_RADIUS * scale));
        Template[] templates = makeTemplates(scale, radius);
        int minPixels = Math.max(30, Math.round(config.minArrowPixels * scale * scale));

        float threshold = 3.4f * scale;
        float thresholdSq = threshold * threshold;
        int total = roiW * roiH;
        for (int i = 0; i < total; i++) {
            headMask[i] = distSq[i] >= thresholdSq ? (byte) 1 : (byte) 0;
        }
        headBlobs.label(headMask, roiW, roiH, true);
        int minHeadPixels = config.minHeadPixels > 0
                ? config.minHeadPixels : Math.max(4, Math.round(6f * scale * scale));
        List<Head> heads = new ArrayList<>();
        for (int component = 1; component <= headBlobs.count; component++) {
            if (headBlobs.area[component] < minHeadPixels) {
                continue;
            }
            int bx = headBlobs.x[component];
            int by = headBlobs.y[component];
            int bw = headBlobs.w[component];
            int bh = headBlobs.h[component];
            float best = -1f;
            int bestX = bx;
            int bestY = by;
            for (int y = by; y < by + bh; y++) {
                int row = y * roiW;
                for (int x = bx; x < bx + bw; x++) {
                    if (headBlobs.labels[row + x] != component) {
                        continue;
                    }
                    float value = distSq[row + x];
                    if (value > best) {
                        best = value;
                        bestX = x;
                        bestY = y;
                    }
                }
            }
            heads.add(new Head(bestX, bestY, (float) Math.sqrt(best)));
        }
        if (heads.isEmpty()) {
            return new ArrayList<>();
        }

        blobs.label(mask, roiW, roiH, true);
        Map<Integer, List<Head>> perBlob = new HashMap<>();
        for (Head head : heads) {
            int blob = blobs.labels[head.y * roiW + head.x];
            if (blob != 0) {
                perBlob.computeIfAbsent(blob, key -> new ArrayList<>()).add(head);
            }
        }

        List<Arrow> arrows = new ArrayList<>();
        float[] centeredPatch = new float[(2 * radius + 1) * (2 * radius + 1)];
        for (Map.Entry<Integer, List<Head>> entry : perBlob.entrySet()) {
            int blob = entry.getKey();
            List<Head> seeds = entry.getValue();
            int[][][] parts;
            if (seeds.size() > 1) {
                parts = splitComponent(blob, seeds);
            } else {
                int[][] pixels = componentPixels(blob);
                parts = new int[][][]{{pixels[0], pixels[1]}};
            }
            for (int index = 0; index < seeds.size(); index++) {
                Head head = seeds.get(index);
                int[] xs = parts[index][0];
                int[] ys = parts[index][1];
                if (xs.length < minPixels) {
                    continue;
                }
                float[] patch = ownPatch(xs, ys, head.x, head.y, radius);
                float[] scores = new float[4];
                char direction = vote(patch, templates, scores, centeredPatch);
                float best = -Float.MAX_VALUE;
                float second = -Float.MAX_VALUE;
                for (float score : scores) {
                    if (score > best) {
                        second = best;
                        best = score;
                    } else if (score > second) {
                        second = score;
                    }
                }
                int[] absoluteXs = offset(xs, roiX);
                int[] absoluteYs = offset(ys, roiY);
                arrows.add(new Arrow(head.x + roiX, head.y + roiY, direction,
                        best - second, scale, absoluteXs, absoluteYs, scores));
            }
        }
        return arrows;
    }

    private int[][][] splitComponent(int blob, List<Head> seeds) {
        int bx = blobs.x[blob];
        int by = blobs.y[blob];
        int bw = blobs.w[blob];
        int bh = blobs.h[blob];
        int[] owner = new int[bw * bh];
        int[] queue = new int[bw * bh];
        int tail = 0;
        for (int index = 0; index < seeds.size(); index++) {
            Head head = seeds.get(index);
            int local = (head.y - by) * bw + (head.x - bx);
            owner[local] = index + 1;
            queue[tail++] = local;
        }
        int head = 0;
        while (head < tail) {
            int current = queue[head++];
            int value = owner[current];
            int y = current / bw;
            int x = current - y * bw;
            if (x > 0) {
                tail = grow(current - 1, value, blob, bx, by, bw, bh, owner, queue, tail);
            }
            if (x + 1 < bw) {
                tail = grow(current + 1, value, blob, bx, by, bw, bh, owner, queue, tail);
            }
            if (y > 0) {
                tail = grow(current - bw, value, blob, bx, by, bw, bh, owner, queue, tail);
            }
            if (y + 1 < bh) {
                tail = grow(current + bw, value, blob, bx, by, bw, bh, owner, queue, tail);
            }
        }

        int[] counts = new int[seeds.size()];
        for (int value : owner) {
            if (value > 0) {
                counts[value - 1]++;
            }
        }
        int[][][] parts = new int[seeds.size()][][];
        for (int index = 0; index < seeds.size(); index++) {
            parts[index] = new int[][]{new int[counts[index]], new int[counts[index]]};
        }
        int[] fill = new int[seeds.size()];
        for (int local = 0; local < owner.length; local++) {
            int value = owner[local];
            if (value == 0) {
                continue;
            }
            int part = value - 1;
            int y = local / bw;
            int x = local - y * bw;
            parts[part][0][fill[part]] = bx + x;
            parts[part][1][fill[part]] = by + y;
            fill[part]++;
        }
        for (int index = 0; index < seeds.size(); index++) {
            if (parts[index][0].length == 0) {
                parts[index][0] = new int[]{seeds.get(index).x};
                parts[index][1] = new int[]{seeds.get(index).y};
            }
        }
        return parts;
    }

    private int grow(int local, int value, int blob, int bx, int by, int bw, int bh,
                     int[] owner, int[] queue, int tail) {
        int y = local / bw;
        int x = local - y * bw;
        int global = (by + y) * roiW + bx + x;
        if (owner[local] == 0 && blobs.labels[global] == blob) {
            owner[local] = value;
            queue[tail++] = local;
        }
        return tail;
    }

    private int[][] componentPixels(int blob) {
        int bx = blobs.x[blob];
        int by = blobs.y[blob];
        int bw = blobs.w[blob];
        int bh = blobs.h[blob];
        int count = 0;
        for (int y = by; y < by + bh; y++) {
            int row = y * roiW;
            for (int x = bx; x < bx + bw; x++) {
                if (blobs.labels[row + x] == blob) {
                    count++;
                }
            }
        }
        int[] xs = new int[count];
        int[] ys = new int[count];
        int index = 0;
        for (int y = by; y < by + bh; y++) {
            int row = y * roiW;
            for (int x = bx; x < bx + bw; x++) {
                if (blobs.labels[row + x] == blob) {
                    xs[index] = x;
                    ys[index] = y;
                    index++;
                }
            }
        }
        return new int[][]{xs, ys};
    }

    private static int[] offset(int[] values, int delta) {
        int[] out = new int[values.length];
        for (int i = 0; i < values.length; i++) {
            out[i] = values[i] + delta;
        }
        return out;
    }

    private static float[] ownPatch(int[] xs, int[] ys, int centerX, int centerY, int radius) {
        int size = 2 * radius + 1;
        float[] patch = new float[size * size];
        for (int i = 0; i < xs.length; i++) {
            int dx = xs[i] - centerX + radius;
            int dy = ys[i] - centerY + radius;
            if (dx >= 0 && dx < size && dy >= 0 && dy < size) {
                patch[dy * size + dx] = 1.0f;
            }
        }
        return patch;
    }

    private static char vote(float[] patch, Template[] templates, float[] scores,
                             float[] centeredPatch) {
        double mean = 0.0;
        for (float value : patch) {
            mean += value;
        }
        mean /= patch.length;
        double norm = 0.0;
        for (int i = 0; i < patch.length; i++) {
            double centered = patch[i] - mean;
            centeredPatch[i] = (float) centered;
            norm += centered * centered;
        }
        norm = Math.sqrt(norm);
        if (norm < 1e-6) {
            return 'U';
        }
        char bestDirection = 'U';
        double best = -2.0;
        double second = -2.0;
        for (int direction = 0; direction < 4; direction++) {
            Template template = templates[direction];
            double dot = 0.0;
            for (int i = 0; i < patch.length; i++) {
                dot += template.centered[i] * centeredPatch[i];
            }
            double score = template.norm > 0.0 ? dot / (norm * template.norm) : 0.0;
            scores[direction] = (float) score;
            if (score > best) {
                second = best;
                best = score;
                bestDirection = Arrow.DIRECTIONS[direction];
            } else if (score > second) {
                second = score;
            }
        }
        return bestDirection;
    }

    private static Template[] makeTemplates(float scale, int radius) {
        int size = 2 * radius + 1;
        byte[] up = triangle(scale, radius, size);
        Template[] result = new Template[4];
        result[0] = centred(up);
        result[1] = centred(rotate(up, size, 1));
        result[2] = centred(rotate(up, size, 2));
        result[3] = centred(rotate(up, size, 3));
        return result;
    }

    private static byte[] triangle(float scale, int radius, int size) {
        byte[] art = new byte[size * size];
        float apex = radius - HEAD_APEX_OFFSET * scale;
        float base = apex + HEAD_HEIGHT * scale;
        float half = HEAD_BASE * scale / 2.0f;
        int apexX = radius;
        int apexY = Math.round(apex);
        int baseY = Math.round(base);
        int leftX = Math.round(radius - half);
        int rightX = Math.round(radius + half);
        fillTriangle(art, size, apexX, apexY, leftX, rightX, baseY);

        int stubHalf = Math.max(1, Math.round(SHAFT_HALF_WIDTH * scale));
        int stubEnd = Math.min(size - 1, Math.round(base + SHAFT_STUB * scale));
        if (stubEnd > baseY) {
            int y0 = Math.max(0, baseY);
            int y1 = Math.min(size - 1, stubEnd);
            int x0 = Math.max(0, radius - stubHalf);
            int x1 = Math.min(size - 1, radius + stubHalf);
            for (int y = y0; y <= y1; y++) {
                for (int x = x0; x <= x1; x++) {
                    art[y * size + x] = 1;
                }
            }
        }
        return art;
    }

    private static void fillTriangle(byte[] art, int size, int apexX, int apexY,
                                     int leftX, int rightX, int baseY) {
        int top = Math.min(apexY, baseY);
        int bottom = Math.max(apexY, baseY);
        for (int y = top; y <= bottom; y++) {
            double t = bottom == apexY ? 0.0 : (y - apexY) / (double) (baseY - apexY);
            // triangle points: apex, (leftX,baseY), (rightX,baseY)
            double left = apexX + (leftX - apexX) * t;
            double right = apexX + (rightX - apexX) * t;
            int x0 = (int) Math.ceil(Math.min(left, right) - 1e-6);
            int x1 = (int) Math.floor(Math.max(left, right) + 1e-6);
            x0 = Math.max(0, x0);
            x1 = Math.min(size - 1, x1);
            for (int x = x0; x <= x1; x++) {
                art[y * size + x] = 1;
            }
        }
    }

    private static byte[] rotate(byte[] source, int size, int quarterTurns) {
        byte[] out = new byte[source.length];
        for (int y = 0; y < size; y++) {
            for (int x = 0; x < size; x++) {
                int sx;
                int sy;
                switch (quarterTurns & 3) {
                    case 1: // 90 degrees clockwise
                        sx = y;
                        sy = size - 1 - x;
                        break;
                    case 2:
                        sx = size - 1 - x;
                        sy = size - 1 - y;
                        break;
                    case 3:
                        sx = size - 1 - y;
                        sy = x;
                        break;
                    default:
                        sx = x;
                        sy = y;
                        break;
                }
                out[y * size + x] = source[sy * size + sx];
            }
        }
        return out;
    }

    private static Template centred(byte[] art) {
        float[] centered = new float[art.length];
        double mean = 0.0;
        for (byte value : art) {
            mean += value;
        }
        mean /= art.length;
        double norm = 0.0;
        for (int i = 0; i < art.length; i++) {
            centered[i] = (float) (art[i] - mean);
            norm += centered[i] * centered[i];
        }
        return new Template(centered, Math.sqrt(norm));
    }

    /** Inclusive (x0,y0,x1,y1) pixel box from screen fractions. */
    public static int[] fractionsToBox(float[] fractions, int width, int height) {
        if (fractions == null || fractions.length != 4) {
            return new int[]{0, 0, width, height};
        }
        int x0 = clamp((int) Math.round(fractions[0] * width), 0, width - 1);
        int y0 = clamp((int) Math.round(fractions[1] * height), 0, height - 1);
        int x1 = clamp((int) Math.round(fractions[2] * width), x0 + 1, width);
        int y1 = clamp((int) Math.round(fractions[3] * height), y0 + 1, height);
        return new int[]{x0, y0, x1, y1};
    }

    private static int clamp(int value, int low, int high) {
        return Math.max(low, Math.min(high, value));
    }

    private static final class Head {
        final int x;
        final int y;
        final float thickness;

        Head(int x, int y, float thickness) {
            this.x = x;
            this.y = y;
            this.thickness = thickness;
        }
    }

    private static final class Template {
        final float[] centered;
        final double norm;

        Template(float[] centered, double norm) {
            this.centered = centered;
            this.norm = norm;
        }
    }
}
