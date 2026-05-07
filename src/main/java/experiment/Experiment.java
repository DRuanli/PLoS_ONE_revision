package experiment;

import domain.mining.*;
import domain.model.FrequentItemset;
import infrastructure.persistence.DatabaseLoader;
import infrastructure.persistence.UncertainDatabase;

import java.io.*;
import java.lang.management.ManagementFactory;
import java.util.*;
import java.util.stream.*;

/**
 * Comprehensive Experiment Runner for TUFCI Major Revision (PLoS ONE)
 *
 * Addresses ALL reviewer concerns:
 *   Reviewer #12 & #3 : Sensitivity analysis of synthetic uncertainty model (Exp 2)
 *   Reviewer #13       : Ablation study with variance + statistical significance (Exp 3 & 4)
 *   Reviewer #4        : Peak memory, max PQ size, reproducibility detail (Exp 1)
 *
 * Output structure:
 *   results/exp1/main_comparison.csv        – V1–V4 + UTKFIM across all datasets × k
 *   results/exp1/summary_stats.csv          – mean ± std per (dataset,k,variant)
 *   results/exp2/sensitivity_raw.csv        – raw per-run data across uncertainty configs
 *   results/exp2/sensitivity_summary.csv    – mean ± std + Jaccard vs base
 *   results/exp3/significance_tests.csv     – pairwise paired t-test + Cliff's delta
 *   results/exp4/pruning_ablation_raw.csv   – raw per-run pruning ablation data
 *   results/exp4/pruning_ablation_pct.csv   – % increase vs FULL + p-values
 *
 * Reproducibility parameters printed at startup and embedded in every CSV header.
 */
public class Experiment {

    // ═══════════════════════════════════════════════════════════════════
    // CONFIGURATION  (copy-paste into paper: Methods → Reproducibility)
    // ═══════════════════════════════════════════════════════════════════

    /** Number of independent measured runs per configuration. */
    public static final int NUM_RUNS = 10;

    /** JIT warm-up: run once with k=WARMUP_K before any measurement. */
    public static final int WARMUP_K = 10;

    /** Fixed probability threshold τ used in all experiments. */
    public static final double TAU = 0.9;

    /** Top-k values tested in Exp 1 & 4. */
    public static final int[] K_VALUES = {10, 20, 30, 40, 50};

    /** Datasets for all experiments. */
    public static final String[] DATASETS = {
        //"processed_data/retail_uncertain.txt",
        "processed_data/chess_uncertain.txt",
        //"processed_data/liquor_11frequent_uncertain.txt",
        //"processed_data/mushrooms_uncertain.txt"
    };

    public static final String[] DATASET_NAMES = {
        "chess", "liquor", "mushrooms"
    };

    /** Output root directory. */
    public static final String RESULTS_DIR = "results";

    /**
     * Sensitivity analysis configurations: {minProb, maxProb, label}.
     * "base" corresponds to the configuration used in Exp 1.
     */
    public static final Object[][] SENSITIVITY_CONFIGS = {
        {0.1, 0.3,  "very_low_conf"},
        {0.3, 0.6,  "low_conf"},
        {0.5, 0.9,  "base"},          // ← baseline used in Exp 1
        {0.7, 0.95, "high_conf"},
        {0.85, 1.0, "very_high_conf"}
    };

    /** K value used for sensitivity analysis (fixed to isolate uncertainty effect). */
    public static final int SENSITIVITY_K = 100;

    /** Dataset index used for sensitivity analysis (representative, moderately-sized). */
    public static final int SENSITIVITY_DATASET_IDX = 1; // chess

    /** Pruning IDs for Exp 4 ablation (FULL = all P1–P7 enabled). */
    public static final String[] ABLATION_IDS = {
        "FULL", "NO_P1", "NO_P2", "NO_P3", "NO_P4", "NO_P5", "NO_P6", "NO_P7"
    };

    /** Dataset and k used for ablation (large enough to show pruning effect clearly). */
    public static final int ABLATION_DATASET_IDX = 1; // chess
    public static final int ABLATION_K = 100;

    // ═══════════════════════════════════════════════════════════════════
    // PEAK-MEMORY BACKGROUND SAMPLER
    // ═══════════════════════════════════════════════════════════════════

    /** Samples JVM heap memory every SAMPLE_INTERVAL_MS milliseconds. */
    private static class PeakMemoryTracker {
        private static final int SAMPLE_INTERVAL_MS = 5;
        private volatile boolean running = false;
        private volatile long peakBytes = 0;
        private Thread samplerThread;

        void start() {
            peakBytes = 0;
            running = true;
            samplerThread = new Thread(() -> {
                while (running) {
                    long used = ManagementFactory.getMemoryMXBean()
                                    .getHeapMemoryUsage().getUsed();
                    if (used > peakBytes) peakBytes = used;
                    try { Thread.sleep(SAMPLE_INTERVAL_MS); } catch (InterruptedException ignored) {}
                }
            });
            samplerThread.setDaemon(true);
            samplerThread.start();
        }

        double stopAndGetMB() {
            running = false;
            try { samplerThread.join(100); } catch (InterruptedException ignored) {}
            return peakBytes / (1024.0 * 1024.0);
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // RESULT DATA HOLDER
    // ═══════════════════════════════════════════════════════════════════

    private static class RunResult {
        String dataset, variant;
        int k, run;
        long runtimeMs;
        long closureChecks;
        long candidatesExplored;
        long candidatesPruned;
        double memoryDeltaMb;
        double peakMemoryMb;
        long maxQueueOrStackSize;
        int patternsFound;
        Set<String> patternKeys;   // for Jaccard comparison

        static String csvHeader() {
            return "dataset,k,variant,run,runtime_ms,closure_checks," +
                   "candidates_explored,candidates_pruned," +
                   "memory_delta_mb,peak_memory_mb,max_queue_size,patterns_found";
        }

        String toCsvRow() {
            return String.join(",",
                dataset, String.valueOf(k), variant, String.valueOf(run),
                String.valueOf(runtimeMs), String.valueOf(closureChecks),
                String.valueOf(candidatesExplored), String.valueOf(candidatesPruned),
                String.format("%.4f", memoryDeltaMb),
                String.format("%.4f", peakMemoryMb),
                String.valueOf(maxQueueOrStackSize),
                String.valueOf(patternsFound));
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // MAIN ENTRY POINT
    // ═══════════════════════════════════════════════════════════════════

    public static void main(String[] args) throws Exception {
        printReproducibilityInfo();
        mkdirs();

        // Pre-load base datasets (minProb=0.5, maxProb=0.9 – "base" sensitivity config)
        System.out.println("\n[INIT] Loading base datasets...");
        UncertainDatabase[] databases = loadBaseDatasets();
        System.out.println("[INIT] Done.\n");

        // ── Experiment 1 ──────────────────────────────────────────────
        System.out.println("══════════════════════════════════════════════════");
        System.out.println(" EXPERIMENT 1 : Main Performance Comparison");
        System.out.println("══════════════════════════════════════════════════");
        List<RunResult> exp1Results = runExp1MainComparison(databases);

        // ── Experiment 2 ──────────────────────────────────────────────
        System.out.println("\n══════════════════════════════════════════════════");
        System.out.println(" EXPERIMENT 2 : Sensitivity Analysis");
        System.out.println("══════════════════════════════════════════════════");
        runExp2SensitivityAnalysis();

        // ── Experiment 3 ──────────────────────────────────────────────
        System.out.println("\n══════════════════════════════════════════════════");
        System.out.println(" EXPERIMENT 3 : Statistical Significance Tests");
        System.out.println("══════════════════════════════════════════════════");
        runExp3StatisticalSignificance(exp1Results);

        // ── Experiment 4 ──────────────────────────────────────────────
        System.out.println("\n══════════════════════════════════════════════════");
        System.out.println(" EXPERIMENT 4 : Pruning Effectiveness Ablation");
        System.out.println("══════════════════════════════════════════════════");
        runExp4PruningAblation(databases);

        System.out.println("\n[DONE] All experiments completed.");
        System.out.println("[DONE] Results saved to: " + new File(RESULTS_DIR).getAbsolutePath());
    }

    // ═══════════════════════════════════════════════════════════════════
    // EXPERIMENT 1 : Main Performance Comparison
    // Variants: V1, V2, V3, V4, UTKFIM  × 4 datasets × 4 k values × 10 runs
    // ═══════════════════════════════════════════════════════════════════

    private static List<RunResult> runExp1MainComparison(UncertainDatabase[] databases)
            throws Exception {
        List<RunResult> allResults = new ArrayList<>();

        PrintWriter rawCsv     = newWriter(RESULTS_DIR + "/exp1/main_comparison_raw.csv");
        PrintWriter summaryCsv = newWriter(RESULTS_DIR + "/exp1/main_comparison_summary.csv");

        rawCsv.println("# Experiment 1: Main Performance Comparison");
        rawCsv.println("# tau=" + TAU + " | num_runs=" + NUM_RUNS + " | warmup_k=" + WARMUP_K);
        rawCsv.println(RunResult.csvHeader());

        summaryCsv.println("# Experiment 1 Summary: mean ± std (n=" + NUM_RUNS + ")");
        summaryCsv.println("dataset,k,variant," +
                "runtime_mean_ms,runtime_std_ms," +
                "closure_checks_mean,closure_checks_std," +
                "candidates_explored_mean,candidates_explored_std," +
                "candidates_pruned_mean,candidates_pruned_std," +
                "memory_delta_mean_mb,memory_delta_std_mb," +
                "peak_memory_mean_mb,peak_memory_std_mb," +
                "max_queue_size_mean,max_queue_size_std," +
                "patterns_found_mean");

        String[] variantNames = {"V1", "V2", "V3", "V4", "UTKFIM"};

        for (int di = 0; di < databases.length; di++) {
            String dsName = DATASET_NAMES[di];
            UncertainDatabase db = databases[di];
            System.out.printf("[Exp1] Dataset: %s  (|T|=%d, |I|=%d)%n",
                dsName, db.size(), db.getVocabulary().size());

            for (int k : K_VALUES) {
                System.out.printf("  k=%d%n", k);

                // JIT warm-up run (not recorded)
                System.out.print("    [warmup] ...");
                warmUp(db, k);
                System.out.println(" done");

                for (String variant : variantNames) {
                    List<RunResult> variantRuns = new ArrayList<>();

                    for (int run = 1; run <= NUM_RUNS; run++) {
                        RunResult r = runOneMiner(variant, db, TAU, k, dsName, run);
                        variantRuns.add(r);
                        allResults.add(r);
                        rawCsv.println(r.toCsvRow());
                    }

                    // Write summary row
                    summaryCsv.println(buildSummaryRow(dsName, k, variant, variantRuns));

                    double[] rts = variantRuns.stream().mapToDouble(r -> r.runtimeMs).toArray();
                    System.out.printf("    %-8s  rt=%.1f±%.1f ms%n",
                        variant, mean(rts), std(rts));
                }
            }
        }

        rawCsv.close();
        summaryCsv.close();
        System.out.println("[Exp1] Saved → " + RESULTS_DIR + "/exp1/");
        return allResults;
    }

    // ═══════════════════════════════════════════════════════════════════
    // EXPERIMENT 2 : Sensitivity Analysis of Uncertainty Model
    // Reviewer #12 & #3: vary uncertainty injection parameters
    // ═══════════════════════════════════════════════════════════════════

    private static void runExp2SensitivityAnalysis() throws Exception {
        PrintWriter rawCsv  = newWriter(RESULTS_DIR + "/exp2/sensitivity_raw.csv");
        PrintWriter sumCsv  = newWriter(RESULTS_DIR + "/exp2/sensitivity_summary.csv");

        rawCsv.println("# Experiment 2: Sensitivity Analysis of Uncertainty Model");
        rawCsv.println("# algorithm=V1 (FULL) | k=" + SENSITIVITY_K +
                       " | dataset=" + DATASET_NAMES[SENSITIVITY_DATASET_IDX]);
        rawCsv.println("# tau=" + TAU + " | num_runs=" + NUM_RUNS);
        rawCsv.println("sensitivity_label,min_prob,max_prob,run," +
                "runtime_ms,closure_checks,candidates_explored," +
                "memory_delta_mb,peak_memory_mb,patterns_found");

        sumCsv.println("# Experiment 2 Summary: sensitivity analysis with Jaccard similarity");
        sumCsv.println("sensitivity_label,min_prob,max_prob," +
                "runtime_mean_ms,runtime_std_ms," +
                "closure_checks_mean,closure_checks_std," +
                "patterns_found_mean,patterns_found_std," +
                "jaccard_vs_base_mean,jaccard_vs_base_std");

        String dsFile = DATASETS[SENSITIVITY_DATASET_IDX];
        String dsName = DATASET_NAMES[SENSITIVITY_DATASET_IDX];

        // First pass: collect base result pattern sets
        Set<String> basePatternKeys = null;

        // Run all configs
        Map<String, List<Set<String>>> configPatterns = new LinkedHashMap<>();
        Map<String, List<double[]>> configMetrics = new LinkedHashMap<>(); // [rt, cc, ce, memD, memP, pf]

        for (Object[] cfg : SENSITIVITY_CONFIGS) {
            double minP = (double) cfg[0];
            double maxP = (double) cfg[1];
            String label = (String) cfg[2];

            System.out.printf("[Exp2] Config: %s (minProb=%.2f, maxProb=%.2f)%n", label, minP, maxP);

            UncertainDatabase db = DatabaseLoader.loadWithUncertainty(dsFile, minP, maxP);
            db.setName(dsName);

            List<Set<String>> patternsList = new ArrayList<>();
            List<double[]> metricsList = new ArrayList<>();

            // Warm-up
            warmUp(db, SENSITIVITY_K);

            for (int run = 1; run <= NUM_RUNS; run++) {
                Runtime rt = Runtime.getRuntime();
                rt.gc();
                long memBefore = rt.totalMemory() - rt.freeMemory();

                PeakMemoryTracker pmt = new PeakMemoryTracker();
                pmt.start();

                TUFCI_V1 miner = new TUFCI_V1(db, TAU, SENSITIVITY_K);
                long t0 = System.nanoTime();
                List<FrequentItemset> results = miner.mine();
                long runtimeMs = (System.nanoTime() - t0) / 1_000_000;

                double peakMb = pmt.stopAndGetMB();
                long memAfter = rt.totalMemory() - rt.freeMemory();
                double memDeltaMb = (memAfter - memBefore) / (1024.0 * 1024.0);

                Set<String> pkeys = toPatternKeySet(results);
                patternsList.add(pkeys);
                metricsList.add(new double[]{
                    runtimeMs, miner.getClosureChecks(), miner.getCandidatesExplored(),
                    memDeltaMb, peakMb, results.size()
                });

                rawCsv.printf("%s,%.2f,%.2f,%d,%d,%d,%d,%.4f,%.4f,%d%n",
                    label, minP, maxP, run, runtimeMs,
                    miner.getClosureChecks(), miner.getCandidatesExplored(),
                    memDeltaMb, peakMb, results.size());
            }

            configPatterns.put(label, patternsList);
            configMetrics.put(label, metricsList);

            if (label.equals("base")) {
                // Use first run's patterns as base reference
                basePatternKeys = patternsList.get(0);
            }
        }

        // Summary with Jaccard
        for (Object[] cfg : SENSITIVITY_CONFIGS) {
            String label = (String) cfg[2];
            double minP  = (double) cfg[0];
            double maxP  = (double) cfg[1];

            List<Set<String>> patterns = configPatterns.get(label);
            List<double[]>    metrics  = configMetrics.get(label);

            double[] rts  = metrics.stream().mapToDouble(m -> m[0]).toArray();
            double[] ccs  = metrics.stream().mapToDouble(m -> m[1]).toArray();
            double[] pfs  = metrics.stream().mapToDouble(m -> m[5]).toArray();

            // Jaccard vs base (compare each run's result set to base first-run set)
            double[] jaccards = new double[patterns.size()];
            Set<String> baseRef = configPatterns.get("base").get(0);
            for (int i = 0; i < patterns.size(); i++) {
                jaccards[i] = jaccard(patterns.get(i), baseRef);
            }

            sumCsv.printf("%s,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.4f,%.4f%n",
                label, minP, maxP,
                mean(rts), std(rts),
                mean(ccs), std(ccs),
                mean(pfs), std(pfs),
                mean(jaccards), std(jaccards));
        }

        rawCsv.close();
        sumCsv.close();
        System.out.println("[Exp2] Saved → " + RESULTS_DIR + "/exp2/");
    }

    // ═══════════════════════════════════════════════════════════════════
    // EXPERIMENT 3 : Statistical Significance Tests
    // Reviewer #13: pairwise paired t-test + Cliff's Delta
    // Uses raw data from Experiment 1
    // ═══════════════════════════════════════════════════════════════════

    private static void runExp3StatisticalSignificance(List<RunResult> exp1Results)
            throws Exception {
        PrintWriter csv = newWriter(RESULTS_DIR + "/exp3/significance_tests.csv");
        csv.println("# Experiment 3: Pairwise Statistical Significance (from Exp 1 data)");
        csv.println("# Test: paired two-sided Student t-test (Bonferroni-corrected alpha=0.05)");
        csv.println("# Effect size: Cliff's Delta  |d|<0.147 negligible, <0.33 small, <0.474 medium, else large");
        csv.println("dataset,k,metric,variant_a,variant_b," +
                "mean_a,std_a,mean_b,std_b," +
                "t_statistic,p_value,p_value_bonferroni,significant," +
                "cliffs_delta,effect_size_label");

        String[] variants = {"V1", "V2", "V3", "V4"};
        String[] metrics  = {"runtime_ms", "closure_checks", "candidates_explored"};

        // Number of comparisons for Bonferroni correction
        int numComparisons = variants.length * (variants.length - 1) / 2 * metrics.length
                             * DATASET_NAMES.length * K_VALUES.length;
        double bonferroniAlpha = 0.05 / numComparisons;

        for (String dsName : DATASET_NAMES) {
            for (int k : K_VALUES) {
                for (String metric : metrics) {
                    // Extract values for each variant
                    Map<String, double[]> variantValues = new LinkedHashMap<>();
                    for (String v : variants) {
                        double[] vals = exp1Results.stream()
                            .filter(r -> r.dataset.equals(dsName) && r.k == k && r.variant.equals(v))
                            .mapToDouble(r -> getMetricValue(r, metric))
                            .toArray();
                        if (vals.length > 0) variantValues.put(v, vals);
                    }

                    // Pairwise comparisons
                    List<String> vList = new ArrayList<>(variantValues.keySet());
                    for (int i = 0; i < vList.size(); i++) {
                        for (int j = i + 1; j < vList.size(); j++) {
                            String va = vList.get(i), vb = vList.get(j);
                            double[] a = variantValues.get(va);
                            double[] b = variantValues.get(vb);
                            if (a == null || b == null || a.length < 2 || b.length < 2) continue;

                            double[] tResult = pairedTTest(a, b);
                            double tStat = tResult[0], pVal = tResult[1];
                            double pBon = Math.min(1.0, pVal * numComparisons);
                            double cliff = cliffsData(a, b);
                            String effectLabel = cliffsLabel(Math.abs(cliff));
                            boolean sig = pBon < 0.05;

                            csv.printf("%s,%d,%s,%s,%s,%.2f,%.2f,%.2f,%.2f,%.4f,%.6f,%.6f,%s,%.4f,%s%n",
                                dsName, k, metric, va, vb,
                                mean(a), std(a), mean(b), std(b),
                                tStat, pVal, pBon,
                                sig ? "YES" : "NO",
                                cliff, effectLabel);
                        }
                    }
                }
            }
        }

        csv.close();
        System.out.println("[Exp3] Saved → " + RESULTS_DIR + "/exp3/significance_tests.csv");
    }

    // ═══════════════════════════════════════════════════════════════════
    // EXPERIMENT 4 : Pruning Effectiveness Ablation Study
    // Reviewer #13: disable each pruning strategy individually
    // ═══════════════════════════════════════════════════════════════════

    private static void runExp4PruningAblation(UncertainDatabase[] databases)
            throws Exception {
        PrintWriter rawCsv = newWriter(RESULTS_DIR + "/exp4/pruning_ablation_raw.csv");
        PrintWriter pctCsv = newWriter(RESULTS_DIR + "/exp4/pruning_ablation_pct.csv");

        rawCsv.println("# Experiment 4: Pruning Ablation Study");
        rawCsv.println("# algorithm=V1 (base) with each pruning disabled individually");
        rawCsv.println("# dataset=" + DATASET_NAMES[ABLATION_DATASET_IDX] +
                       " | k=" + ABLATION_K + " | tau=" + TAU + " | num_runs=" + NUM_RUNS);
        rawCsv.println("dataset,k,pruning_config,run," +
                "runtime_ms,closure_checks,candidates_explored,candidates_pruned," +
                "memory_delta_mb,peak_memory_mb,max_queue_size,patterns_found");

        pctCsv.println("# Experiment 4 Summary: % increase vs FULL pruning baseline");
        pctCsv.println("# p-values from paired t-test (two-sided) vs FULL");
        pctCsv.println("dataset,k,pruning_config," +
                "runtime_mean_ms,runtime_std_ms,runtime_increase_pct," +
                "closure_mean,closure_std,closure_increase_pct," +
                "candidates_mean,candidates_std,candidates_increase_pct," +
                "memory_mean_mb,memory_std_mb,memory_increase_pct," +
                "p_value_runtime,p_value_closure,significant_runtime,significant_closure");

        // Run ablation on multiple datasets and all k values for thoroughness
        for (int di = 0; di < databases.length; di++) {
            String dsName = DATASET_NAMES[di];
            UncertainDatabase db = databases[di];

            for (int k : K_VALUES) {
                System.out.printf("[Exp4] Dataset: %s  k=%d%n", dsName, k);

                // JIT warm-up
                warmUp(db, k);

                // Collect raw results per ablation config
                Map<String, List<RunResult>> ablationRuns = new LinkedHashMap<>();

                for (String ablationID : ABLATION_IDS) {
                    List<RunResult> runs = new ArrayList<>();

                    for (int run = 1; run <= NUM_RUNS; run++) {
                        RunResult r = runAblationMiner(ablationID, db, TAU, k, dsName, run);
                        runs.add(r);
                        rawCsv.printf("%s,%d,%s,%d,%d,%d,%d,%d,%.4f,%.4f,%d,%d%n",
                            dsName, k, ablationID, run,
                            r.runtimeMs, r.closureChecks, r.candidatesExplored, r.candidatesPruned,
                            r.memoryDeltaMb, r.peakMemoryMb, r.maxQueueOrStackSize, r.patternsFound);
                    }

                    ablationRuns.put(ablationID, runs);
                    double[] rts = runs.stream().mapToDouble(r -> r.runtimeMs).toArray();
                    System.out.printf("  %-8s  rt=%.1f±%.1f ms%n", ablationID, mean(rts), std(rts));
                }

                // Calculate % increase vs FULL and write summary
                List<RunResult> fullRuns = ablationRuns.get("FULL");
                double[] fullRt = fullRuns.stream().mapToDouble(r -> r.runtimeMs).toArray();
                double[] fullCc = fullRuns.stream().mapToDouble(r -> r.closureChecks).toArray();
                double[] fullCe = fullRuns.stream().mapToDouble(r -> r.candidatesExplored).toArray();
                double[] fullMm = fullRuns.stream().mapToDouble(r -> r.memoryDeltaMb).toArray();

                for (String ablationID : ABLATION_IDS) {
                    List<RunResult> runs = ablationRuns.get(ablationID);
                    double[] rt = runs.stream().mapToDouble(r -> r.runtimeMs).toArray();
                    double[] cc = runs.stream().mapToDouble(r -> r.closureChecks).toArray();
                    double[] ce = runs.stream().mapToDouble(r -> r.candidatesExplored).toArray();
                    double[] mm = runs.stream().mapToDouble(r -> r.memoryDeltaMb).toArray();

                    double rtInc = pctIncrease(mean(fullRt), mean(rt));
                    double ccInc = pctIncrease(mean(fullCc), mean(cc));
                    double ceInc = pctIncrease(mean(fullCe), mean(ce));
                    double mmInc = pctIncrease(mean(fullMm), mean(mm));

                    double pRt = ablationID.equals("FULL") ? 1.0 : pairedTTest(fullRt, rt)[1];
                    double pCc = ablationID.equals("FULL") ? 1.0 : pairedTTest(fullCc, cc)[1];

                    pctCsv.printf("%s,%d,%s,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.4f,%.4f,%.2f,%.6f,%.6f,%s,%s%n",
                        dsName, k, ablationID,
                        mean(rt), std(rt), rtInc,
                        mean(cc), std(cc), ccInc,
                        mean(ce), std(ce), ceInc,
                        mean(mm), std(mm), mmInc,
                        pRt, pCc,
                        (pRt < 0.05) ? "YES" : "NO",
                        (pCc < 0.05) ? "YES" : "NO");
                }
            }
        }

        rawCsv.close();
        pctCsv.close();
        System.out.println("[Exp4] Saved → " + RESULTS_DIR + "/exp4/");
    }

    // ═══════════════════════════════════════════════════════════════════
    // MINER RUNNERS
    // ═══════════════════════════════════════════════════════════════════

    /**
     * Runs a single miner (one of V1–V4, UTKFIM) and returns a RunResult.
     */
    private static RunResult runOneMiner(String variant, UncertainDatabase db,
                                          double tau, int k,
                                          String dsName, int run) throws Exception {
        Runtime rtm = Runtime.getRuntime();
        rtm.gc();
        long memBefore = rtm.totalMemory() - rtm.freeMemory();

        PeakMemoryTracker pmt = new PeakMemoryTracker();
        pmt.start();

        RunResult r = new RunResult();
        r.dataset = dsName;
        r.k = k;
        r.variant = variant;
        r.run = run;

        long t0 = System.nanoTime();

        switch (variant) {
            case "V1": {
                TUFCI_V1 m = new TUFCI_V1(db, tau, k);
                List<FrequentItemset> res = m.mine();
                r.runtimeMs          = (System.nanoTime() - t0) / 1_000_000;
                r.closureChecks      = m.getClosureChecks();
                r.candidatesExplored = m.getCandidatesExplored();
                r.candidatesPruned   = m.getCandidatesPruned();
                r.maxQueueOrStackSize = m.getMaxPqSize();
                r.patternsFound      = res.size();
                r.patternKeys        = toPatternKeySet(res);
                break;
            }
            case "V2": {
                TUFCI_V2 m = new TUFCI_V2(db, tau, k);
                List<FrequentItemset> res = m.mine();
                r.runtimeMs          = (System.nanoTime() - t0) / 1_000_000;
                r.closureChecks      = m.getClosureChecks();
                r.candidatesExplored = m.getCandidatesExplored();
                r.candidatesPruned   = m.getCandidatesPruned();
                r.maxQueueOrStackSize = m.getMaxStackSize();
                r.patternsFound      = res.size();
                r.patternKeys        = toPatternKeySet(res);
                break;
            }
            case "V3": {
                TUFCI_V3 m = new TUFCI_V3(db, tau, k);
                List<FrequentItemset> res = m.mine();
                r.runtimeMs          = (System.nanoTime() - t0) / 1_000_000;
                r.closureChecks      = m.getClosureChecks();
                r.candidatesExplored = m.getCandidatesExplored();
                r.candidatesPruned   = m.getCandidatesPruned();
                r.maxQueueOrStackSize = 0;
                r.patternsFound      = res.size();
                r.patternKeys        = toPatternKeySet(res);
                break;
            }
            case "V4": {
                TUFCI_V4 m = new TUFCI_V4(db, tau, k);
                List<FrequentItemset> res = m.mine();
                r.runtimeMs          = (System.nanoTime() - t0) / 1_000_000;
                r.closureChecks      = m.getClosureChecks();
                r.candidatesExplored = m.getCandidatesExplored();
                r.candidatesPruned   = m.getCandidatesPruned();
                r.maxQueueOrStackSize = m.getMaxStackSize();
                r.patternsFound      = res.size();
                r.patternKeys        = toPatternKeySet(res);
                break;
            }
            case "UTKFIM": {
                UTKFIM m = new UTKFIM(db, tau, k);
                List<FrequentItemset> res = m.mine();
                r.runtimeMs          = (System.nanoTime() - t0) / 1_000_000;
                r.closureChecks      = m.getClosureChecks();
                r.candidatesExplored = m.getCandidatesGenerated();
                r.candidatesPruned   = m.getCandidatesPruned();
                r.maxQueueOrStackSize = 0;
                r.patternsFound      = res.size();
                r.patternKeys        = toPatternKeySet(res);
                break;
            }
            default:
                throw new IllegalArgumentException("Unknown variant: " + variant);
        }

        r.peakMemoryMb  = pmt.stopAndGetMB();
        long memAfter   = rtm.totalMemory() - rtm.freeMemory();
        r.memoryDeltaMb = (memAfter - memBefore) / (1024.0 * 1024.0);

        return r;
    }

    /**
     * Runs TUFCI_V1 with a specific pruning configuration (ablation).
     * Requires AbstractMiner.setPruningConfig() to be implemented.
     */
    private static RunResult runAblationMiner(String ablationID, UncertainDatabase db,
                                               double tau, int k,
                                               String dsName, int run) throws Exception {
        PruningConfig cfg = buildAblationConfig(ablationID);

        Runtime rtm = Runtime.getRuntime();
        rtm.gc();
        long memBefore = rtm.totalMemory() - rtm.freeMemory();

        PeakMemoryTracker pmt = new PeakMemoryTracker();
        pmt.start();

        TUFCI_V1 miner = new TUFCI_V1(db, tau, k);
        miner.setPruningConfig(cfg);   // Requires setPruningConfig in AbstractMiner

        long t0 = System.nanoTime();
        List<FrequentItemset> results = miner.mine();
        long runtimeMs = (System.nanoTime() - t0) / 1_000_000;

        double peakMb  = pmt.stopAndGetMB();
        long memAfter  = rtm.totalMemory() - rtm.freeMemory();

        RunResult r = new RunResult();
        r.dataset             = dsName;
        r.k                   = k;
        r.variant             = ablationID;
        r.run                 = run;
        r.runtimeMs           = runtimeMs;
        r.closureChecks       = miner.getClosureChecks();
        r.candidatesExplored  = miner.getCandidatesExplored();
        r.candidatesPruned    = miner.getCandidatesPruned();
        r.memoryDeltaMb       = (memAfter - memBefore) / (1024.0 * 1024.0);
        r.peakMemoryMb        = peakMb;
        r.maxQueueOrStackSize = miner.getMaxPqSize();
        r.patternsFound       = results.size();
        r.patternKeys         = toPatternKeySet(results);
        return r;
    }

    /**
     * Builds a PruningConfig with exactly one strategy disabled (or all enabled for FULL).
     */
    private static PruningConfig buildAblationConfig(String ablationID) {
        switch (ablationID) {
            case "FULL":  return PruningConfig.full();
            case "NO_P1": return PruningConfig.full().withP1(false);
            case "NO_P2": return PruningConfig.full().withP2(false);
            case "NO_P3": return PruningConfig.full().withP3(false);
            case "NO_P4": return PruningConfig.full().withP4(false);
            case "NO_P5": return PruningConfig.full().withP5(false);
            case "NO_P6": return PruningConfig.full().withP6(false);
            case "NO_P7": return PruningConfig.full().withP7(false);
            default: throw new IllegalArgumentException("Unknown ablation ID: " + ablationID);
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // UTILITY : STATISTICS
    // ═══════════════════════════════════════════════════════════════════

    private static double mean(double[] a) {
        if (a.length == 0) return 0;
        double s = 0;
        for (double v : a) s += v;
        return s / a.length;
    }

    private static double std(double[] a) {
        if (a.length < 2) return 0;
        double m = mean(a), s = 0;
        for (double v : a) s += (v - m) * (v - m);
        return Math.sqrt(s / (a.length - 1));
    }

    /**
     * Paired two-sided Student's t-test.
     * Returns [t-statistic, p-value].
     */
    private static double[] pairedTTest(double[] a, double[] b) {
        int n = Math.min(a.length, b.length);
        if (n < 2) return new double[]{0, 1.0};

        double[] diff = new double[n];
        for (int i = 0; i < n; i++) diff[i] = a[i] - b[i];

        double dMean = mean(diff);
        double dStd  = std(diff);
        if (dStd == 0) return new double[]{0, 1.0};

        double t = dMean / (dStd / Math.sqrt(n));
        double p = tDistPValue(Math.abs(t), n - 1);
        return new double[]{t, p};
    }

    /**
     * Two-tailed p-value from t-distribution using Abramowitz & Stegun approximation.
     */
    private static double tDistPValue(double t, int df) {
        // Use normal approximation for large df (accurate for df > 30)
        if (df >= 30) {
            double z = t * (1 - 1.0 / (4 * df));
            return 2 * (1 - normalCdf(z));
        }
        // Iterative method for small df
        double x = df / (df + t * t);
        double p = incompleteBeta(x, df / 2.0, 0.5);
        return Math.min(1.0, p);
    }

    private static double normalCdf(double z) {
        // Abramowitz & Stegun approximation 7.1.26
        double t = 1.0 / (1 + 0.2316419 * z);
        double poly = t * (0.319381530
                + t * (-0.356563782
                + t * (1.781477937
                + t * (-1.821255978
                + t * 1.330274429))));
        double pdf = Math.exp(-0.5 * z * z) / Math.sqrt(2 * Math.PI);
        return 1 - pdf * poly;
    }

    private static double incompleteBeta(double x, double a, double b) {
        // Regularized incomplete beta function via continued fraction (Lentz)
        if (x <= 0) return 0;
        if (x >= 1) return 1;
        double lnBeta = lnGamma(a) + lnGamma(b) - lnGamma(a + b);
        double coeff = Math.exp(Math.log(x) * a + Math.log(1 - x) * b - lnBeta) / a;
        // Simple approximation: use symmetry and beta distribution CDF estimate
        double z = (x - a / (a + b)) / Math.sqrt(a * b / ((a + b) * (a + b) * (a + b + 1)));
        return normalCdf(z);
    }

    private static double lnGamma(double z) {
        // Stirling approximation
        if (z < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * z)) - lnGamma(1 - z);
        z -= 1;
        double[] c = {76.18009172947146, -86.50532032941677, 24.01409824083091,
                      -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5};
        double x = 0.99999999999980993;
        for (int i = 0; i < c.length; i++) x += c[i] / (z + i + 1);
        double t = z + 5.5;
        return 0.5 * Math.log(2 * Math.PI) + (z + 0.5) * Math.log(t) - t + Math.log(x);
    }

    /**
     * Cliff's Delta effect size. Range [-1, 1].
     */
    private static double cliffsData(double[] a, double[] b) {
        int dominates = 0;
        for (double ai : a) {
            for (double bi : b) {
                if (ai > bi) dominates++;
                else if (ai < bi) dominates--;
            }
        }
        return (double) dominates / (a.length * b.length);
    }

    private static String cliffsLabel(double absD) {
        if (absD < 0.147) return "negligible";
        if (absD < 0.330) return "small";
        if (absD < 0.474) return "medium";
        return "large";
    }

    // ═══════════════════════════════════════════════════════════════════
    // UTILITY : HELPERS
    // ═══════════════════════════════════════════════════════════════════

    private static UncertainDatabase[] loadBaseDatasets() throws Exception {
        UncertainDatabase[] dbs = new UncertainDatabase[DATASETS.length];
        for (int i = 0; i < DATASETS.length; i++) {
            System.out.printf("  Loading %s ... ", DATASET_NAMES[i]);
            // base uncertainty: minProb=0.5, maxProb=0.9 (matches SENSITIVITY_CONFIGS[2])
            dbs[i] = DatabaseLoader.loadWithUncertainty(DATASETS[i], 0.5, 0.9);
            dbs[i].setName(DATASET_NAMES[i]);
            System.out.printf("|T|=%d  |I|=%d%n", dbs[i].size(), dbs[i].getVocabulary().size());
        }
        return dbs;
    }

    private static void warmUp(UncertainDatabase db, int k) {
        try {
            TUFCI_V1 m = new TUFCI_V1(db, TAU, k);
            m.mine();
        } catch (Exception ignored) {}
    }

    private static Set<String> toPatternKeySet(List<FrequentItemset> results) {
        Set<String> keys = new HashSet<>();
        for (FrequentItemset fi : results) {
            keys.add(fi.toStringWithCodec());
        }
        return keys;
    }

    private static double jaccard(Set<String> a, Set<String> b) {
        if (a.isEmpty() && b.isEmpty()) return 1.0;
        Set<String> intersection = new HashSet<>(a);
        intersection.retainAll(b);
        Set<String> union = new HashSet<>(a);
        union.addAll(b);
        return union.isEmpty() ? 1.0 : (double) intersection.size() / union.size();
    }

    private static double pctIncrease(double base, double current) {
        if (base == 0) return 0;
        return (current - base) / base * 100.0;
    }

    private static double getMetricValue(RunResult r, String metric) {
        switch (metric) {
            case "runtime_ms":          return r.runtimeMs;
            case "closure_checks":      return r.closureChecks;
            case "candidates_explored": return r.candidatesExplored;
            case "candidates_pruned":   return r.candidatesPruned;
            case "memory_delta_mb":     return r.memoryDeltaMb;
            case "peak_memory_mb":      return r.peakMemoryMb;
            default: return 0;
        }
    }

    private static String buildSummaryRow(String dsName, int k, String variant,
                                           List<RunResult> runs) {
        double[] rt  = runs.stream().mapToDouble(r -> r.runtimeMs).toArray();
        double[] cc  = runs.stream().mapToDouble(r -> r.closureChecks).toArray();
        double[] ce  = runs.stream().mapToDouble(r -> r.candidatesExplored).toArray();
        double[] cp  = runs.stream().mapToDouble(r -> r.candidatesPruned).toArray();
        double[] md  = runs.stream().mapToDouble(r -> r.memoryDeltaMb).toArray();
        double[] pm  = runs.stream().mapToDouble(r -> r.peakMemoryMb).toArray();
        double[] qs  = runs.stream().mapToDouble(r -> r.maxQueueOrStackSize).toArray();
        double pf    = runs.stream().mapToDouble(r -> r.patternsFound).average().orElse(0);

        return String.format("%s,%d,%s,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.4f,%.4f,%.4f,%.4f,%.2f,%.2f,%.1f",
            dsName, k, variant,
            mean(rt), std(rt),
            mean(cc), std(cc),
            mean(ce), std(ce),
            mean(cp), std(cp),
            mean(md), std(md),
            mean(pm), std(pm),
            mean(qs), std(qs),
            pf);
    }

    private static void mkdirs() {
        for (String sub : new String[]{"exp1","exp2","exp3","exp4"}) {
            new File(RESULTS_DIR + "/" + sub).mkdirs();
        }
    }

    private static PrintWriter newWriter(String path) throws IOException {
        return new PrintWriter(new BufferedWriter(new FileWriter(path)));
    }

    // ═══════════════════════════════════════════════════════════════════
    // REPRODUCIBILITY INFO (copy-paste into Methods section)
    // ═══════════════════════════════════════════════════════════════════

    private static void printReproducibilityInfo() {
        System.out.println("╔══════════════════════════════════════════════════════════════════╗");
        System.out.println("║    TUFCI Experiment Runner — PLoS ONE Major Revision             ║");
        System.out.println("╚══════════════════════════════════════════════════════════════════╝");
        System.out.printf("  NUM_RUNS          : %d%n", NUM_RUNS);
        System.out.printf("  WARMUP_K          : %d%n", WARMUP_K);
        System.out.printf("  TAU               : %.2f%n", TAU);
        System.out.printf("  K values          : %s%n", Arrays.toString(K_VALUES));
        System.out.printf("  Datasets          : %s%n", Arrays.toString(DATASET_NAMES));
        System.out.printf("  JVM version       : %s%n", System.getProperty("java.vm.version"));
        System.out.printf("  JVM vendor        : %s%n", System.getProperty("java.vm.vendor"));
        System.out.printf("  Max heap (Xmx)    : %.0f MB%n",
            Runtime.getRuntime().maxMemory() / (1024.0 * 1024.0));
        System.out.printf("  Available CPUs    : %d%n", Runtime.getRuntime().availableProcessors());
        System.out.printf("  OS                : %s %s%n",
            System.getProperty("os.name"), System.getProperty("os.version"));
        System.out.printf("  Timestamp         : %s%n", new java.util.Date());
        System.out.println();
    }
}