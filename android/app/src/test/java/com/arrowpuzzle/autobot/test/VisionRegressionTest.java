package com.arrowpuzzle.autobot.test;

import com.arrowpuzzle.autobot.core.Arrow;
import com.arrowpuzzle.autobot.core.Config;
import com.arrowpuzzle.autobot.core.Counter;
import com.arrowpuzzle.autobot.core.Detection;
import com.arrowpuzzle.autobot.core.Detector;
import com.arrowpuzzle.autobot.core.Frame;
import com.arrowpuzzle.autobot.core.Solver;
import org.junit.Test;

import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

/**
 * Proves the Java port still reads the real screenshots exactly like the Python
 * pipeline that was validated on the phone.  Regenerate reference.tsv with
 * {@code python tools/export_android_reference.py} after shared logic changes.
 */
public class VisionRegressionTest {
    @Test
    public void readsEveryRealBoardExactly() throws Exception {
        Map<String, Expected> expected = readReference();
        assertEquals(3, expected.size());
        Config config = new Config().validate();
        Detector detector = new Detector(config);
        int checked = 0;
        for (Map.Entry<String, Expected> entry : expected.entrySet()) {
            Expected want = entry.getValue();
            String label = entry.getKey();
            String fixture = label.substring(0, label.length() - ".png".length())
                    + ".argb.z";
            Frame frame;
            try (InputStream stream = required("/samples/" + fixture)) {
                frame = TestFrames.load(stream);
            }

            Detection detection = detector.detect(frame);
            assertEquals(label + " arrow count", want.count, detection.arrows.size());
            assertEquals(label + " scale", want.scale, detection.scale, 0.05f);
            assertEquals(label + " left edge", want.x0, detection.x0, 2);
            assertEquals(label + " top edge", want.y0, detection.y0, 2);
            assertEquals(label + " right edge", want.x1, detection.x1, 2);
            assertEquals(label + " bottom edge", want.y1, detection.y1, 2);

            Integer counter = Counter.read(frame, config);
            assertNotNull(label + " HUD counter", counter);
            assertEquals(label + " HUD counter", (int) want.counter, (int) counter);

            List<Arrow> order = Solver.order(detection.arrows);
            assertNotNull(label + " clear plan", order);
            assertEquals(label + " plan length", want.plan, order.size());
            assertPlanIsSafe(label, detection.arrows, order);

            Set<Arrow> used = new HashSet<>();
            for (ExpectedArrow arrow : want.arrows) {
                Arrow match = null;
                for (Arrow candidate : detection.arrows) {
                    if (!used.contains(candidate)
                            && Math.abs(candidate.headX - arrow.x) <= 2
                            && Math.abs(candidate.headY - arrow.y) <= 2) {
                        match = candidate;
                        break;
                    }
                }
                assertNotNull(label + " missing head at " + arrow.x + "," + arrow.y, match);
                used.add(match);
                assertEquals(label + " direction at " + arrow.x + "," + arrow.y,
                        arrow.direction, match.direction);
            }
            checked += detection.arrows.size();
        }
        assertEquals(208, checked);
    }

    private static void assertPlanIsSafe(String label, List<Arrow> arrows, List<Arrow> order) {
        List<Arrow> alive = new ArrayList<>(arrows);
        for (Arrow arrow : order) {
            assertTrue(label + " unsafe tap " + arrow, Solver.isSafe(arrow, alive));
            assertTrue(alive.remove(arrow));
        }
        assertTrue(label + " plan left arrows", alive.isEmpty());
    }

    private static Map<String, Expected> readReference() throws Exception {
        InputStream stream = required("/reference.tsv");
        Map<String, Expected> expected = new LinkedHashMap<>();
        boolean arrowsSection = false;
        try (BufferedReader reader = new BufferedReader(
                new InputStreamReader(stream, StandardCharsets.UTF_8))) {
            String line;
            while ((line = reader.readLine()) != null) {
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
                    Expected item = expected.computeIfAbsent(parts[0], key -> new Expected());
                    item.arrows.add(new ExpectedArrow(Integer.parseInt(parts[1]),
                            Integer.parseInt(parts[2]), parts[3].charAt(0)));
                }
            }
        }
        return expected;
    }

    private static InputStream required(String resource) {
        InputStream stream = VisionRegressionTest.class.getResourceAsStream(resource);
        if (stream == null) {
            throw new IllegalStateException("missing test resource " + resource);
        }
        return stream;
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
