package experiment;

import domain.mining.TUFCI_V1;
import domain.model.FrequentItemset;
import domain.model.Itemset;
import infrastructure.persistence.UncertainDatabase;

import java.util.*;

/**
 * Exp3 — Uncertainty parameter sensitivity analysis.
 *
 * Reviewer mapping:
 *   - Reviewer #1 #12 (CRITICAL): "synthetic uncertainty arbitrary, no
 *     sensitivity analysis"
 *   - Reviewer #2 #3 (CRITICAL): "robustness of injected uncertainty model"
 *
 * Methodology:
 *   For each parameter (alpha, rho, P_min, P_max), vary it across a sweep
 *   while holding the other three at the reference value. For each setting,
 *   run TUFCI_V1 and record:
 *     - Top-k result set
 *     - Runtime
 *   Then compute Jaccard similarity of top-k itemsets vs the reference run.
 *
 * Acceptable stability: Jaccard >= 0.85 across reasonable parameter range
 * indicates that conclusions are not artifacts of the specific synthetic
 * injection parameters.
 *
 * Output:
 *   results/exp3/sensitivity_raw.csv — one row per (dataset, param, value, rep)
 *   results/exp3/sensitivity_jaccard.csv — Jaccard vs reference, per setting
 *
 * Datasets: Chess (dense), Retail (sparse). Two datasets are sufficient for
 * sensitivity analysis given the high cost of full grid sweeps.
 *
 * @author Le, Vo, Nguyen
 */
public class Exp3_UncertaintySensitivity {

    private static final String OUT_DIR = "results/exp3";
    private static final double TAU = 0.7;
    private static final int K = 10;
    private static final int WARMUP = 1;
    private static final int REPS = 3;

    private static final String[] BASE_DATASETS = {
            "chess_uncertain",
            //"liquor_11frequent_uncertain",
            //"mushrooms_uncertain",
            //"retail_uncertain"
    };

    private static final double[] ALPHA_VALUES = { 0.5, 1.0, 1.5, 2.0 };
    private static final double[] RHO_VALUES   = { 0.0, 0.02, 0.05, 0.10 };
    private static final double[] PMIN_VALUES  = { 0.05, 0.1, 0.2, 0.3 };
    private static final double[] PMAX_VALUES  = { 0.7, 0.8, 0.9, 1.0 };

    public static void main(String[] args) throws Exception {
        ExperimentRunner.ensureDir(OUT_DIR);

        for (String baseName : BASE_DATASETS) {
            System.out.println("====== " + baseName + " ======");

            System.out.println("  loading reference top-k...");
            String refFile = String.format(Locale.US, "processed_data/%s_alpha_%.1f.txt", baseName, 1.0);
            UncertainDatabase refDb = UncertainDatabase.loadFromFile(refFile);
            Set<String> refTopK = runOnceGetTopK(refDb);

            try (ExperimentRunner.CsvWriter raw = new ExperimentRunner.CsvWriter(
                    OUT_DIR + "/" + baseName + "_sensitivity_raw.csv", Arrays.asList(
                            "dataset", "param", "value", "rep", "runtime_ms", "topk_count"));
                 ExperimentRunner.CsvWriter jac = new ExperimentRunner.CsvWriter(
                         OUT_DIR + "/" + baseName + "_sensitivity_jaccard.csv", Arrays.asList(
                                 "dataset", "param", "value",
                                 "jaccard_vs_reference", "runtime_mean_ms"))) {

                sweep(raw, jac, baseName, "alpha", ALPHA_VALUES, refTopK,
                        v -> {
                            try {
                                return UncertainDatabase.loadFromFile(
                                    String.format(Locale.US, "processed_data/%s_alpha_%.1f.txt", baseName, v));
                            } catch (Exception e) {
                                throw new RuntimeException(e);
                            }
                        });

                sweep(raw, jac, baseName, "rho", RHO_VALUES, refTopK,
                        v -> {
                            try {
                                return UncertainDatabase.loadFromFile(
                                    String.format(Locale.US, "processed_data/%s_rho_%.2f.txt", baseName, v));
                            } catch (Exception e) {
                                throw new RuntimeException(e);
                            }
                        });

                sweep(raw, jac, baseName, "p_min", PMIN_VALUES, refTopK,
                        v -> {
                            try {
                                return UncertainDatabase.loadFromFile(
                                    String.format(Locale.US, "processed_data/%s_pmin_%.2f.txt", baseName, v));
                            } catch (Exception e) {
                                throw new RuntimeException(e);
                            }
                        });

                sweep(raw, jac, baseName, "p_max", PMAX_VALUES, refTopK,
                        v -> {
                            try {
                                return UncertainDatabase.loadFromFile(
                                    String.format(Locale.US, "processed_data/%s_pmax_%.1f.txt", baseName, v));
                            } catch (Exception e) {
                                throw new RuntimeException(e);
                            }
                        });
            }
        }

        System.out.println("\n[Exp3] Done. Results in " + OUT_DIR + "/");
    }

    @FunctionalInterface
    private interface DbFactory {
        UncertainDatabase create(double v) throws Exception;
    }

    private static void sweep(ExperimentRunner.CsvWriter raw,
                               ExperimentRunner.CsvWriter jac,
                               String datasetName,
                               String paramName,
                               double[] values,
                               Set<String> refTopK,
                               DbFactory factory) throws Exception {

        for (double v : values) {
            System.out.println("  sweep " + paramName + "=" + v);
            UncertainDatabase db = factory.create(v);

            Set<String> currentTopK = null;
            double[] runtimes = new double[REPS];

            for (int rep = 0; rep < WARMUP + REPS; rep++) {
                TUFCI_V1 m = new TUFCI_V1(db, TAU, K);
                final List<FrequentItemset>[] result = new List[]{ null };
                ExperimentRunner.RunResult r = ExperimentRunner.timedRun(
                        () -> result[0] = m.mine());

                if (rep < WARMUP) continue;
                int repIdx = rep - WARMUP;
                runtimes[repIdx] = r.runtimeMs;

                Map<String, Object> row = new LinkedHashMap<>();
                row.put("dataset", datasetName);
                row.put("param", paramName);
                row.put("value", v);
                row.put("rep", repIdx);
                row.put("runtime_ms", r.runtimeMs);
                row.put("topk_count", result[0] != null ? result[0].size() : 0);
                raw.writeRow(row);

                if (currentTopK == null) currentTopK = canonicalTopK(result[0]);
            }

            ExperimentRunner.Stats stats = ExperimentRunner.Stats.from(runtimes);
            double jaccard = jaccard(refTopK, currentTopK == null ? Collections.emptySet() : currentTopK);

            Map<String, Object> jrow = new LinkedHashMap<>();
            jrow.put("dataset", datasetName);
            jrow.put("param", paramName);
            jrow.put("value", v);
            jrow.put("jaccard_vs_reference", String.format("%.4f", jaccard));
            jrow.put("runtime_mean_ms", String.format("%.2f", stats.mean));
            jac.writeRow(jrow);
        }
    }

    /**
     * Run TUFCI_V1 once on the given database and return the canonicalized
     * top-k set (string-encoded itemsets) for Jaccard comparison.
     */
    private static Set<String> runOnceGetTopK(UncertainDatabase db) {
        TUFCI_V1 m = new TUFCI_V1(db, TAU, K);
        List<FrequentItemset> result = m.mine();
        return canonicalTopK(result);
    }

    /**
     * Convert top-k list to canonical string set: sorted items joined by '_'.
     * This makes results comparable across runs even if probability/support
     * differs slightly due to different uncertainty injections.
     */
    private static Set<String> canonicalTopK(List<FrequentItemset> result) {
        if (result == null) return Collections.emptySet();
        Set<String> set = new HashSet<>();
        for (FrequentItemset fi : result) {
            int[] items = fi.getItemsArray();
            // items are already sorted ASC in our implementation
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < items.length; i++) {
                if (i > 0) sb.append('_');
                sb.append(items[i]);
            }
            set.add(sb.toString());
        }
        return set;
    }

    private static double jaccard(Set<String> a, Set<String> b) {
        if (a.isEmpty() && b.isEmpty()) return 1.0;
        Set<String> inter = new HashSet<>(a);
        inter.retainAll(b);
        Set<String> union = new HashSet<>(a);
        union.addAll(b);
        return inter.size() / (double) union.size();
    }
}