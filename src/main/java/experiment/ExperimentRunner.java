package experiment;

import java.io.*;
import java.nio.file.*;
import java.util.*;

/**
 * ExperimentRunner — Shared utility for running mining experiments with
 * repetition, JVM warm-up, statistics aggregation, and CSV output.
 *
 * All experiment files use this to ensure consistent methodology across runs.
 *
 * Conventions:
 *   - Each repetition is preceded by GC + a 50ms sleep to reduce noise.
 *   - The first WARMUP_REPS runs are discarded (HotSpot warm-up).
 *   - Times are measured in nanoseconds, reported in milliseconds.
 *   - Memory is measured via Runtime.getRuntime() snapshots before/after.
 *
 * @author Le, Vo, Nguyen — PONE-D-26-07832 revision
 */
public class ExperimentRunner {

    public static final int DEFAULT_WARMUP_REPS = 2;
    public static final int DEFAULT_MEASURED_REPS = 5;

    /** A single timed run result. */
    public static class RunResult {
        public long runtimeMs;
        public long peakHeapBytes;
        public Map<String, Object> stats = new LinkedHashMap<>();
    }

    /** Aggregated statistics across multiple repetitions. */
    public static class Stats {
        public double mean;
        public double std;
        public double median;
        public double min;
        public double max;

        public static Stats from(double[] values) {
            Stats s = new Stats();
            if (values.length == 0) return s;
            double sum = 0;
            for (double v : values) sum += v;
            s.mean = sum / values.length;
            double sq = 0;
            for (double v : values) sq += (v - s.mean) * (v - s.mean);
            s.std = (values.length > 1) ? Math.sqrt(sq / (values.length - 1)) : 0.0;
            double[] sorted = values.clone();
            Arrays.sort(sorted);
            s.median = sorted[sorted.length / 2];
            s.min = sorted[0];
            s.max = sorted[sorted.length - 1];
            return s;
        }

        @Override
        public String toString() {
            return String.format("mean=%.2f std=%.2f median=%.2f min=%.2f max=%.2f",
                    mean, std, median, min, max);
        }
    }

    /** Execute a single timed run, capturing runtime and peak heap. */
    public static RunResult timedRun(Runnable mining) {
        Runtime rt = Runtime.getRuntime();

        // Force GC and stabilize
        System.gc();
        try { Thread.sleep(50); } catch (InterruptedException ignore) {}

        long heapBefore = rt.totalMemory() - rt.freeMemory();
        long startNs = System.nanoTime();

        mining.run();

        long endNs = System.nanoTime();
        long heapAfter = rt.totalMemory() - rt.freeMemory();

        RunResult r = new RunResult();
        r.runtimeMs = (endNs - startNs) / 1_000_000;
        r.peakHeapBytes = Math.max(0, heapAfter - heapBefore);
        return r;
    }

    /**
     * Repeat a mining factory and return runtime statistics. The factory is
     * called fresh for each repetition so there is no state leakage.
     *
     * @param factory creates and returns a mining runnable + an extractor for stats
     * @param warmupReps number of warm-up repetitions (discarded)
     * @param measuredReps number of measured repetitions
     * @return list of RunResult, one per measured repetition
     */
    public static List<RunResult> repeatedRuns(
            java.util.function.Supplier<TimedTask> factory,
            int warmupReps, int measuredReps) {

        List<RunResult> measured = new ArrayList<>();

        for (int i = 0; i < warmupReps + measuredReps; i++) {
            TimedTask task = factory.get();
            RunResult r = timedRun(task.runnable);
            if (task.statsExtractor != null) task.statsExtractor.accept(r.stats);
            if (i >= warmupReps) measured.add(r);
        }
        return measured;
    }

    /** A timed task: a runnable plus an optional extractor for algorithm stats. */
    public static class TimedTask {
        public Runnable runnable;
        public java.util.function.Consumer<Map<String, Object>> statsExtractor;
        public TimedTask(Runnable r) { this.runnable = r; }
        public TimedTask(Runnable r, java.util.function.Consumer<Map<String, Object>> ext) {
            this.runnable = r; this.statsExtractor = ext;
        }
    }

    // ========= CSV writing utilities =========

    public static void ensureDir(String dirPath) {
        try {
            Files.createDirectories(Paths.get(dirPath));
        } catch (IOException e) {
            throw new RuntimeException("Could not create dir " + dirPath, e);
        }
    }

    public static class CsvWriter implements Closeable {
        private final BufferedWriter w;
        private final List<String> header;

        public CsvWriter(String path, List<String> header) throws IOException {
            this.w = new BufferedWriter(new FileWriter(path));
            this.header = header;
            w.write(String.join(",", header));
            w.newLine();
            w.flush();
        }

        public void writeRow(Map<String, Object> row) throws IOException {
            StringBuilder sb = new StringBuilder();
            boolean first = true;
            for (String col : header) {
                if (!first) sb.append(",");
                Object v = row.get(col);
                sb.append(v == null ? "" : escape(v.toString()));
                first = false;
            }
            w.write(sb.toString());
            w.newLine();
            w.flush();
        }

        private String escape(String s) {
            if (s.contains(",") || s.contains("\"") || s.contains("\n")) {
                return "\"" + s.replace("\"", "\"\"") + "\"";
            }
            return s;
        }

        @Override public void close() throws IOException { w.close(); }
    }

    /** Compute Wilcoxon signed-rank test (paired) p-value approximation. */
    public static double wilcoxonSignedRank(double[] a, double[] b) {
        if (a.length != b.length) throw new IllegalArgumentException("paired arrays must equal length");
        int n = a.length;
        if (n < 5) return Double.NaN; // insufficient sample size

        double[] diff = new double[n];
        for (int i = 0; i < n; i++) diff[i] = a[i] - b[i];

        // Drop zeros
        int nonZero = 0;
        for (double d : diff) if (d != 0) nonZero++;
        if (nonZero == 0) return 1.0;

        double[] absDiff = new double[nonZero];
        int[] sign = new int[nonZero];
        int idx = 0;
        for (double d : diff) {
            if (d == 0) continue;
            absDiff[idx] = Math.abs(d);
            sign[idx] = d > 0 ? 1 : -1;
            idx++;
        }

        // Rank
        Integer[] order = new Integer[nonZero];
        for (int i = 0; i < nonZero; i++) order[i] = i;
        final double[] absDiffFinal = absDiff;
        Arrays.sort(order, Comparator.comparingDouble(i -> absDiffFinal[i]));
        double[] ranks = new double[nonZero];
        for (int i = 0; i < nonZero; ) {
            int j = i;
            while (j < nonZero && absDiff[order[j]] == absDiff[order[i]]) j++;
            double avgRank = (i + j + 1) / 2.0;  // 1-indexed
            for (int kk = i; kk < j; kk++) ranks[order[kk]] = avgRank;
            i = j;
        }

        double Wplus = 0, Wminus = 0;
        for (int i = 0; i < nonZero; i++) {
            if (sign[i] > 0) Wplus += ranks[i];
            else Wminus += ranks[i];
        }
        double W = Math.min(Wplus, Wminus);

        // Normal approximation
        double mean = nonZero * (nonZero + 1) / 4.0;
        double var = nonZero * (nonZero + 1) * (2.0 * nonZero + 1) / 24.0;
        double z = (W - mean) / Math.sqrt(var);
        // two-sided p-value via standard normal CDF approximation
        return 2.0 * (1.0 - normalCdf(Math.abs(z)));
    }

    /** Standard normal CDF using Abramowitz-Stegun approximation. */
    private static double normalCdf(double x) {
        double t = 1.0 / (1.0 + 0.2316419 * x);
        double d = 0.3989422804 * Math.exp(-x * x / 2);
        double p = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))));
        return 1.0 - p;
    }
}