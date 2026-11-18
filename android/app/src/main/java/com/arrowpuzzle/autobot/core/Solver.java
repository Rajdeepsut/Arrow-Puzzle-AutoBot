package com.arrowpuzzle.autobot.core;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Which arrows can leave the board without touching anything - pure geometry.
 *
 * <p>An arrow may only be tapped when nothing lies in the corridor in front of
 * its head: the half-plane it points into, restricted to a narrow band around
 * its own axis.  Removing an arrow can never put an obstacle into someone
 * else's corridor, so the set of tappable arrows only grows and a greedy sweep
 * is provably optimal.  If it gets stuck, no tap order could have cleared the
 * board with the arrows as read.</p>
 */
public final class Solver {
    /** Half width of the corridor, in pixels at artwork scale 1.0. */
    public static final float CORRIDOR_HALF = 7.0f;

    private Solver() {
    }

    public static boolean isSafe(Arrow arrow, List<Arrow> alive) {
        return isSafe(arrow, alive, CORRIDOR_HALF * arrow.scale, 0f);
    }

    public static boolean isSafe(Arrow arrow, List<Arrow> alive, float half, float gap) {
        int originX = arrow.headX;
        int originY = arrow.headY;
        boolean horizontal = arrow.direction == 'L' || arrow.direction == 'R';
        for (Arrow other : alive) {
            if (other == arrow) {
                continue;
            }
            int[] xs = other.xs;
            int[] ys = other.ys;
            int length = xs.length;
            if (horizontal) {
                float low = originY - half;
                float high = originY + half;
                for (int i = 0; i < length; i++) {
                    float py = ys[i];
                    if (py < low || py > high) {
                        continue;
                    }
                    float px = xs[i];
                    if (arrow.direction == 'R' ? px > originX + gap : px < originX - gap) {
                        return false;
                    }
                }
            } else {
                float low = originX - half;
                float high = originX + half;
                for (int i = 0; i < length; i++) {
                    float px = xs[i];
                    if (px < low || px > high) {
                        continue;
                    }
                    float py = ys[i];
                    if (arrow.direction == 'D' ? py > originY + gap : py < originY - gap) {
                        return false;
                    }
                }
            }
        }
        return true;
    }

    public static List<Arrow> safeArrows(List<Arrow> arrows, float half, float gap) {
        List<Arrow> safe = new ArrayList<>();
        for (Arrow arrow : arrows) {
            if (isSafe(arrow, arrows, half, gap)) {
                safe.add(arrow);
            }
        }
        return safe;
    }

    public static List<Arrow> safeArrows(List<Arrow> arrows) {
        List<Arrow> safe = new ArrayList<>();
        for (Arrow arrow : arrows) {
            if (isSafe(arrow, arrows)) {
                safe.add(arrow);
            }
        }
        return safe;
    }

    /** Greedy clear plan as waves; {@code null} when the board dead-locks. */
    public static List<List<Arrow>> plan(List<Arrow> arrows) {
        List<Arrow> alive = new ArrayList<>(arrows);
        List<List<Arrow>> waves = new ArrayList<>();
        while (!alive.isEmpty()) {
            List<Arrow> wave = safeArrows(alive);
            if (wave.isEmpty()) {
                return null;
            }
            waves.add(wave);
            alive.removeAll(wave);
        }
        return waves;
    }

    /** The same plan as one flat tap order, or {@code null}. */
    public static List<Arrow> order(List<Arrow> arrows) {
        List<List<Arrow>> waves = plan(arrows);
        if (waves == null) {
            return null;
        }
        List<Arrow> order = new ArrayList<>();
        for (List<Arrow> wave : waves) {
            order.addAll(wave);
        }
        return order;
    }

    public static boolean clearable(List<Arrow> arrows) {
        return plan(arrows) != null;
    }

    /**
     * Find the smallest direction correction that makes a deadlocked board clear.
     * Returns {@code {arrow: direction}} for the first assignment that works.
     */
    public static Map<Arrow, Character> repair(List<Arrow> arrows) {
        return repair(arrows, 6, 2, 120);
    }

    public static Map<Arrow, Character> repair(List<Arrow> arrows, int maxAmbiguous,
                                               int maxChanges, int maxChecks) {
        List<Arrow> base = new ArrayList<>(arrows);
        if (base.isEmpty()) {
            return new HashMap<>();
        }
        if (clearable(base)) {
            return new HashMap<>();
        }
        List<Arrow> ranked = new ArrayList<>(base);
        ranked.sort((a, b) -> Float.compare(a.confidence, b.confidence));
        List<Arrow> candidates = ranked.subList(0, Math.min(Math.max(0, maxAmbiguous), ranked.size()));

        List<Arrow> optionArrows = new ArrayList<>();
        List<List<Character>> optionDirections = new ArrayList<>();
        for (Arrow arrow : candidates) {
            List<Character> alternatives = new ArrayList<>();
            for (char direction : Arrow.DIRECTIONS) {
                if (direction != arrow.direction) {
                    alternatives.add(direction);
                }
            }
            if (arrow.scores != null) {
                float best = -Float.MAX_VALUE;
                for (float score : arrow.scores) {
                    best = Math.max(best, score);
                }
                final float floor = best - 0.5f;
                alternatives.sort((a, b) -> Float.compare(
                        arrow.scores[Arrow.indexOf(b)], arrow.scores[Arrow.indexOf(a)]));
                List<Character> credible = new ArrayList<>();
                for (char direction : alternatives) {
                    if (arrow.scores[Arrow.indexOf(direction)] >= floor) {
                        credible.add(direction);
                    }
                }
                alternatives = credible;
            }
            if (!alternatives.isEmpty()) {
                optionArrows.add(arrow);
                optionDirections.add(alternatives);
            }
        }
        if (optionArrows.isEmpty()) {
            return null;
        }

        int[] checks = {0};
        Map<Arrow, Character> found = new HashMap<>();
        int limit = Math.min(optionArrows.size(), Math.max(0, maxChanges));
        for (int changes = 1; changes <= limit; changes++) {
            boolean done = choose(optionArrows, optionDirections, base, changes, 0, 0,
                    new int[changes], found, checks, Math.max(1, maxChecks));
            if (done) {
                return found;
            }
            if (checks[0] >= maxChecks) {
                return null;
            }
        }
        return null;
    }

    private static boolean choose(List<Arrow> optionArrows,
                                  List<List<Character>> optionDirections,
                                  List<Arrow> base, int changes, int start, int depth,
                                  int[] chosen, Map<Arrow, Character> found, int[] checks,
                                  int maxChecks) {
        if (depth == changes) {
            return product(optionArrows, optionDirections, base, chosen, 0, found, checks,
                    maxChecks);
        }
        for (int i = start; i <= optionArrows.size() - (changes - depth); i++) {
            chosen[depth] = i;
            if (choose(optionArrows, optionDirections, base, changes, i + 1, depth + 1,
                    chosen, found, checks, maxChecks)) {
                return true;
            }
            if (checks[0] >= maxChecks) {
                return false;
            }
        }
        return false;
    }

    private static boolean product(List<Arrow> optionArrows,
                                   List<List<Character>> optionDirections,
                                   List<Arrow> base, int[] chosen, int depth,
                                   Map<Arrow, Character> found, int[] checks,
                                   int maxChecks) {
        if (depth == chosen.length) {
            Map<Arrow, Character> fixes = new HashMap<>();
            for (int index : chosen) {
                Character direction = found.get(optionArrows.get(index));
                if (direction != null) {
                    fixes.put(optionArrows.get(index), direction);
                }
            }
            // the product recursion below stores the current pick in `found`
            if (fixes.size() != chosen.length) {
                return false;
            }
            List<Arrow> views = new ArrayList<>(base.size());
            for (Arrow arrow : base) {
                Character direction = fixes.get(arrow);
                views.add(direction == null ? arrow : arrow.withDirection(direction));
            }
            checks[0]++;
            if (clearable(views)) {
                return true;
            }
            return false;
        }
        int optionIndex = chosen[depth];
        for (char direction : optionDirections.get(optionIndex)) {
            found.put(optionArrows.get(optionIndex), direction);
            if (product(optionArrows, optionDirections, base, chosen, depth + 1, found,
                    checks, maxChecks)) {
                return true;
            }
            if (checks[0] >= maxChecks) {
                return false;
            }
        }
        found.remove(optionArrows.get(optionIndex));
        return false;
    }
}
