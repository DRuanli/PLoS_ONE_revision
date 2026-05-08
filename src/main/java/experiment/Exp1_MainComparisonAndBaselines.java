package experiment;

import domain.mining.*;
import domain.model.FrequentItemset;
import experiment.baselines.ITUFP;
import experiment.baselines.TopKPFIM;
import infrastructure.persistence.DatabaseLoader;
import infrastructure.persistence.UncertainDatabase;

import java.io.IOException;
import java.util.*;

/**
 * Exp1 — Main comparison: V1, V2, V3, V4, TopKPFIM, ITUFP on all four datasets.
 *
 * Reviewer mapping:
 *   - Reviewer #2 #2 (CRITICAL): no external baselines  →  TopKPFIM + ITUFP added
 *   - Reviewer #1 #13 (CRITICAL): no variance / significance  →  ≥5 reps + Wilcoxon
 *   - Reviewer #2 #4: PQ size + peak heap statistics      →  exported to CSV
 *
 * Output:
 *   results/exp1/main_comparison_raw.csv      — one row per (dataset, k, algo, rep)
 *   results/exp1/main_comparison_summary.csv  — aggregated mean/std per (dataset, k, algo)
 *   results/exp1/wilcoxon.csv                 — pairwise Wilcoxon V1 vs others on runtime
 *
 * Datasets:    Chess, Mushroom, Retail, Liquor (under datasets/)
 * k values:    {30, 50, 100}
 * tau:         0.7
 * Reps:        2 warmup + 5 measured
 *
 * @author Le, Vo, Nguyen
 */
public class Exp1_MainComparisonAndBaselines {

    private static final String OUT_DIR = "results/exp1";
    private static final double TAU = 0.7;
    private static final int[] K_VALUES = { 10 };
    private static final int WARMUP = 2;
    private static final int REPS = 5;

    private static final String[] DATASETS = {
            "processed_data/chess_uncertain.txt",
            "processed_data/mushrooms_uncertain.txt"
    };

    private static final String[] ALGORITHMS = {
            "V1_BFS_Full", "V2_DFS_Full", "V3_BFS_Search", "V4_DFS_Search",
            "TopKPFIM",
            "ITUFP"
    };

    public static void main(String[] args) throws Exception {
        ExperimentRunner.ensureDir(OUT_DIR);

        List<String> rawHeader = Arrays.asList(
                "dataset", "k", "algorithm", "rep",
                "runtime_ms", "peak_heap_bytes",
                "candidates_explored", "closure_checks",
                "support_calculations", "max_queue_size", "topk_count");

        for (String datasetPath : DATASETS) {
            System.out.println("====== Loading " + datasetPath + " ======");
            UncertainDatabase db;
            try {
                db =  UncertainDatabase.loadFromFile(datasetPath); //DatabaseLoader.loadWithUncertainty(datasetPath, 0.5, 0.9);
            } catch (IOException e) {
                System.err.println("Could not load " + datasetPath + ": " + e.getMessage());
                continue;
            }
            String datasetName = db.getName();

            try (ExperimentRunner.CsvWriter raw = new ExperimentRunner.CsvWriter(
                    OUT_DIR + "/" + datasetName + "_main_comparison_raw.csv", rawHeader);
                 ExperimentRunner.CsvWriter summary = new ExperimentRunner.CsvWriter(
                         OUT_DIR + "/" + datasetName + "_main_comparison_summary.csv", Arrays.asList(
                         "dataset", "k", "algorithm",
                         "runtime_mean_ms", "runtime_std_ms", "runtime_median_ms",
                         "closure_checks_mean", "max_queue_size_mean",
                         "peak_heap_mb_mean"));
                 ExperimentRunner.CsvWriter wilcoxon = new ExperimentRunner.CsvWriter(
                         OUT_DIR + "/" + datasetName + "_wilcoxon.csv", Arrays.asList(
                         "dataset", "k", "comparison", "p_value", "v1_faster"))) {

                for (int k : K_VALUES) {
                    System.out.println("---- " + datasetName + " | k=" + k + " ----");

                    // Track per-algorithm runtime arrays for Wilcoxon
                    Map<String, double[]> runtimesByAlgo = new LinkedHashMap<>();

                    for (String algo : ALGORITHMS) {
                        System.out.println("  algo: " + algo);
                        double[] runtimes = new double[REPS];
                        long closureSum = 0, queueMaxSum = 0, heapSum = 0;
                        int topKSum = 0;

                        for (int rep = 0; rep < WARMUP + REPS; rep++) {
                            ExperimentRunner.RunResult r = runOne(db, algo, k);
                            if (rep < WARMUP) continue;

                            int repIdx = rep - WARMUP;
                            runtimes[repIdx] = r.runtimeMs;
                            closureSum += longStat(r, "closure_checks");
                            queueMaxSum += longStat(r, "max_queue_size");
                            heapSum += r.peakHeapBytes;
                            topKSum += intStat(r, "topk_count");

                            Map<String, Object> row = new LinkedHashMap<>();
                            row.put("dataset", datasetName);
                            row.put("k", k);
                            row.put("algorithm", algo);
                            row.put("rep", repIdx);
                            row.put("runtime_ms", r.runtimeMs);
                            row.put("peak_heap_bytes", r.peakHeapBytes);
                            row.put("candidates_explored", longStat(r, "candidates_explored"));
                            row.put("closure_checks", longStat(r, "closure_checks"));
                            row.put("support_calculations", longStat(r, "support_calculations"));
                            row.put("max_queue_size", longStat(r, "max_queue_size"));
                            row.put("topk_count", intStat(r, "topk_count"));
                            raw.writeRow(row);
                        }

                        runtimesByAlgo.put(algo, runtimes);

                        ExperimentRunner.Stats stats = ExperimentRunner.Stats.from(runtimes);
                        Map<String, Object> srow = new LinkedHashMap<>();
                        srow.put("dataset", datasetName);
                        srow.put("k", k);
                        srow.put("algorithm", algo);
                        srow.put("runtime_mean_ms", String.format("%.2f", stats.mean));
                        srow.put("runtime_std_ms", String.format("%.2f", stats.std));
                        srow.put("runtime_median_ms", String.format("%.2f", stats.median));
                        srow.put("closure_checks_mean", closureSum / (double) REPS);
                        srow.put("max_queue_size_mean", queueMaxSum / (double) REPS);
                        srow.put("peak_heap_mb_mean",
                                String.format("%.2f", heapSum / (double) REPS / 1024.0 / 1024.0));
                        summary.writeRow(srow);
                    }

                    // Wilcoxon: V1 vs each other
                    double[] v1 = runtimesByAlgo.get("V1_BFS_Full");
                    if (v1 != null) {
                        for (String algo : ALGORITHMS) {
                            if (algo.equals("V1_BFS_Full")) continue;
                            double[] other = runtimesByAlgo.get(algo);
                            if (other == null) continue;
                            double p = ExperimentRunner.wilcoxonSignedRank(v1, other);
                            double v1Mean = mean(v1);
                            double otherMean = mean(other);
                            Map<String, Object> wrow = new LinkedHashMap<>();
                            wrow.put("dataset", datasetName);
                            wrow.put("k", k);
                            wrow.put("comparison", "V1_vs_" + algo);
                            wrow.put("p_value", String.format("%.6f", p));
                            wrow.put("v1_faster", v1Mean < otherMean);
                            wilcoxon.writeRow(wrow);
                        }
                    }
                }
            }
        }

        System.out.println("\n[Exp1] Done. Results in " + OUT_DIR + "/");
    }

    /** Run a single algorithm one time and capture stats. */
    private static ExperimentRunner.RunResult runOne(UncertainDatabase db, String algo, int k) {
        ExperimentRunner.TimedTask task = makeTask(db, algo, k);
        ExperimentRunner.RunResult r = ExperimentRunner.timedRun(task.runnable);
        if (task.statsExtractor != null) task.statsExtractor.accept(r.stats);
        return r;
    }

    /** Factory: returns a TimedTask for the given algorithm. */
    private static ExperimentRunner.TimedTask makeTask(UncertainDatabase db, String algo, int k) {
        switch (algo) {
            case "V1_BFS_Full": {
                TUFCI_V1 m = new TUFCI_V1(db, TAU, k);
                final List<FrequentItemset>[] result = new List[]{ null };
                Runnable r = () -> result[0] = m.mine();
                return new ExperimentRunner.TimedTask(r, stats -> {
                    stats.put("candidates_explored", m.getCandidatesExplored());
                    stats.put("closure_checks", m.getClosureChecks());
                    stats.put("support_calculations", m.getSupportCalculations());
                    stats.put("max_queue_size", (long) m.getMaxPqSize());
                    stats.put("topk_count", result[0] != null ? result[0].size() : 0);
                });
            }
            case "V2_DFS_Full": {
                TUFCI_V2 m = new TUFCI_V2(db, TAU, k);
                final List<FrequentItemset>[] result = new List[]{ null };
                Runnable r = () -> result[0] = m.mine();
                return new ExperimentRunner.TimedTask(r, stats -> {
                    stats.put("candidates_explored", m.getCandidatesExplored());
                    stats.put("closure_checks", m.getClosureChecks());
                    stats.put("support_calculations", m.getSupportCalculations());
                    stats.put("max_queue_size", m.getMaxStackSize());
                    stats.put("topk_count", result[0] != null ? result[0].size() : 0);
                });
            }
            case "V3_BFS_Search": {
                TUFCI_V3 m = new TUFCI_V3(db, TAU, k);
                final List<FrequentItemset>[] result = new List[]{ null };
                Runnable r = () -> result[0] = m.mine();
                return new ExperimentRunner.TimedTask(r, stats -> {
                    stats.put("candidates_explored", m.getCandidatesExplored());
                    stats.put("closure_checks", m.getClosureChecks());
                    stats.put("support_calculations", m.getSupportCalculations());
                    stats.put("max_queue_size", 0L); // V3 doesn't track this
                    stats.put("topk_count", result[0] != null ? result[0].size() : 0);
                });
            }
            case "V4_DFS_Search": {
                TUFCI_V4 m = new TUFCI_V4(db, TAU, k);
                final List<FrequentItemset>[] result = new List[]{ null };
                Runnable r = () -> result[0] = m.mine();
                return new ExperimentRunner.TimedTask(r, stats -> {
                    stats.put("candidates_explored", m.getCandidatesExplored());
                    stats.put("closure_checks", m.getClosureChecks());
                    stats.put("support_calculations", m.getSupportCalculations());
                    stats.put("max_queue_size", m.getMaxStackSize());
                    stats.put("topk_count", result[0] != null ? result[0].size() : 0);
                });
            }
            case "TopKPFIM": {
                TopKPFIM m = new TopKPFIM(db, TAU, k);
                final List<FrequentItemset>[] result = new List[]{ null };
                Runnable r = () -> result[0] = m.mine();
                return new ExperimentRunner.TimedTask(r, stats -> {
                    stats.put("candidates_explored", m.getCandidatesExplored());
                    stats.put("closure_checks", m.getClosureChecks());
                    stats.put("support_calculations", m.getSupportCalculations());
                    stats.put("max_queue_size", m.getMaxBufferSize());
                    stats.put("topk_count", result[0] != null ? result[0].size() : 0);
                });
            }
            case "ITUFP": {
                ITUFP m = new ITUFP(db, TAU, k);
                final List<FrequentItemset>[] result = new List[]{ null };
                Runnable r = () -> result[0] = m.mine();
                return new ExperimentRunner.TimedTask(r, stats -> {
                    stats.put("candidates_explored", m.getCandidatesExplored());
                    stats.put("closure_checks", m.getClosureChecks());
                    stats.put("support_calculations", m.getSupportCalculations());
                    stats.put("max_queue_size", m.getMaxStackSize());
                    stats.put("topk_count", result[0] != null ? result[0].size() : 0);
                });
            }
            default:
                throw new IllegalArgumentException("Unknown algorithm: " + algo);
        }
    }

    private static long longStat(ExperimentRunner.RunResult r, String key) {
        Object v = r.stats.get(key);
        if (v instanceof Long) return (Long) v;
        if (v instanceof Integer) return ((Integer) v).longValue();
        return 0L;
    }

    private static int intStat(ExperimentRunner.RunResult r, String key) {
        Object v = r.stats.get(key);
        if (v instanceof Integer) return (Integer) v;
        if (v instanceof Long) return ((Long) v).intValue();
        return 0;
    }

    private static double mean(double[] a) {
        double sum = 0;
        for (double v : a) sum += v;
        return a.length == 0 ? 0 : sum / a.length;
    }
}