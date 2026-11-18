package com.arrowpuzzle.autobot.test;

import com.arrowpuzzle.autobot.core.Arrow;
import com.arrowpuzzle.autobot.core.Config;
import com.arrowpuzzle.autobot.core.Counter;
import com.arrowpuzzle.autobot.core.Detection;
import com.arrowpuzzle.autobot.core.Detector;
import com.arrowpuzzle.autobot.core.Frame;
import com.arrowpuzzle.autobot.core.Solver;

import java.io.File;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Plain-JVM regression check against the Python pipeline's exact reading.
 * Run from android/: {@code java -cp ... com.arrowpuzzle.autobot.test.CoreCheck ..}
 */
public final class CoreCheck {
    private CoreCheck() {
    }

    public static void main(String[] args) throws Exception {
        File root = new File(args.length > 0 ? args[0] : "..");
        File samples = new File(root, "android/app/src/test/resources/samples");
        File reference = new File(root,
                "android/app/src/test/resources/reference.tsv");
        Map<String, Expected> expected = readReference(reference);
        Config config = new Config().validate();
        Detector detector = new Detector(config);
        int failures = 0;
        for (Map.Entry<String, Expected> entry : expected.entrySet()) {
            String name = entry.getKey();
            Expected want = entry.getValue();
            String fixture = name.substring(0, name.length() - ".png".length())
                    + ".argb.z";
            Frame frame;
            try (InputStream stream = Files.newInputStream(
                    new File(samples, fixture).toPath())) {
                frame = TestFrames.load(stream);
            }
            long started = System.nanoTime();
            Detection detection = detector.detect(frame);
            long millis = (System.nanoTime() - started) / 1_000_000;
            Integer counter = Counter.read(frame, config);
            List<Arrow> order = Solver.order(detection.arrows);
            System.out.printf("%s: %d arrows (want %d), counter %s (want %d), "
                            + "plan %s (want %d), scale %.4f (want %.4f), bbox %d,%d,%d,%d "
                            + "(%d,%d,%d,%d), %d ms%n",
                    name, detection.arrows.size(), want.count, counter, want.counter,
                    order == null ? "null" : order.size(), want.plan, detection.scale,
                    want.scale, detection.x0, detection.y0, detection.x1, detection.y1,
                    want.x0, want.y0, want.x1, want.y1, millis);

            if (detection.arrows.size() != want.count) {
                failures++;
            }
            if (counter != null && counter != want.counter) {
                failures++;
            }
            if (order == null || order.size() != want.plan) {
                failures++;
            }
            if (Math.abs(detection.scale - want.scale) > 0.05f) {
                failures++;
            }
            if (Math.abs(detection.x0 - want.x0) > 2 || Math.abs(detection.y0 - want.y0) > 2
                    || Math.abs(detection.x1 - want.x1) > 2
                    || Math.abs(detection.y1 - want.y1) > 2) {
                failures++;
            }
            Set<Arrow> used = new HashSet<>();
            for (ExpectedArrow arrow : want.arrows) {
                Arrow match = null;
                for (Arrow candidate : detection.arrows) {
                    if (used.contains(candidate)) {
                        continue;
                    }
                    if (Math.abs(candidate.headX - arrow.x) <= 2
                            && Math.abs(candidate.headY - arrow.y) <= 2) {
                        match = candidate;
                        break;
                    }
                }
                if (match == null) {
                    System.out.printf("  MISSING head (%d,%d) %c%n", arrow.x, arrow.y,
                            arrow.direction);
                    failures++;
                    continue;
                }
                used.add(match);
                if (match.direction != arrow.direction) {
                    System.out.printf("  WRONG direction at (%d,%d): java %c, python %c, "
                                    + "scores %.2f %.2f %.2f %.2f%n",
                            arrow.x, arrow.y, match.direction, arrow.direction,
                            match.scores[0], match.scores[1], match.scores[2], match.scores[3]);
                    failures++;
                }
            }
            if (order != null && !order.isEmpty()) {
                List<Arrow> alive = new ArrayList<>(detection.arrows);
                for (Arrow arrow : order) {
                    if (!Solver.isSafe(arrow, alive)) {
                        System.out.println("  UNSAFE tap in plan: " + arrow);
                        failures++;
                        break;
                    }
                    alive.remove(arrow);
                }
                if (!alive.isEmpty()) {
                    System.out.println("  PLAN left " + alive.size() + " arrows");
                    failures++;
                }
            }
        }
        if (failures > 0) {
            System.out.println("FAILED: " + failures + " difference(s)");
            System.exit(1);
        }
        System.out.println("OK: Java core matches the Python reference");
    }

    private static Map<String, Expected> readReference(File file) throws Exception {
        Map<String, Expected> expected = new LinkedHashMap<>();
        boolean arrowsSection = false;
        for (String line : Files.readAllLines(file.toPath(), StandardCharsets.UTF_8)) {
            if (line.isBlank()) {
                continue;
            }
            if (line.startsWith("# arrows:")) {
                arrowsSection = true;
                continue;
            }
            if (line.startsWith("#")) {
                continue;
            }
            String[] parts = line.split("\t");
            if (!arrowsSection) {
                Expected item = new Expected();
                item.count = Integer.parseInt(parts[1]);
                item.counter = Integer.parseInt(parts[2]);
                item.plan = Integer.parseInt(parts[3]);
                item.scale = Float.parseFloat(parts[4]);
                item.x0 = Integer.parseInt(parts[5]);
                item.y0 = Integer.parseInt(parts[6]);
                item.x1 = Integer.parseInt(parts[7]);
                item.y1 = Integer.parseInt(parts[8]);
                expected.put(parts[0], item);
            } else {
                Expected item = expected.get(parts[0]);
                if (item == null) {
                    throw new IllegalStateException("arrow for unknown sample " + parts[0]);
                }
                item.arrows.add(new ExpectedArrow(Integer.parseInt(parts[1]),
                        Integer.parseInt(parts[2]), parts[3].charAt(0)));
            }
        }
        return expected;
    }

    private static final class Expected {
        int count;
        int counter;
        int plan;
        float scale;
        int x0;
        int y0;
        int x1;
        int y1;
        final List<ExpectedArrow> arrows = new ArrayList<>();
    }

    private static final class ExpectedArrow {
        final int x;
        final int y;
        final char direction;

        ExpectedArrow(int x, int y, char direction) {
            this.x = x;
            this.y = y;
            this.direction = direction;
        }
    }
}
