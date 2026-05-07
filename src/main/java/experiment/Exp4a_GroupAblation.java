package experiment;

import domain.mining.PruningConfig;
import domain.mining.TUFCI_V1;
import domain.model.FrequentItemset;
import infrastructure.persistence.UncertainDatabase;

import java.util.*;

/**
 * Exp4a — Group ablation study.
 *
 * Reviewer mapping:
 *   - Reviewer #1 #11: pruning strategies presented incrementally without
 *     justification of mechanism
 *   - Reviewer #1 #13: no variance / significance testing in ablation
 *
 * Pruning-group taxonomy:
 *   G1 — Search-frontier termination       : P1 (Phase-2 break) + P2 (Phase-3 break)
 *   G2 — Item-level admissibility          : P3 (item-loop break on support)
 *   G3 — Upper-bound tightening            : P4 (subset bound) + P5 (UB filter)
 *   G4 — Tidset-level shortcuts            : P6 (tidset size) + P7 (closure skip)
 *
 * Configurations:
 *   FULL          — all groups enabled (= V1 reference)
 *   NO_G1         — disable P1 + P2
 *   NO_G2         — disable P3
 *   NO_G3         — disable P4 + P5
 *   NO_G4         — disable P6 + P7
 *   ONLY_G1       — only frontier termination (others off)
 *   ONLY_G2_G3_G4 — pruning without best-first (=approximates V3-equivalent on cost shortcuts)
 *
 * Reps: ≥5 measured + Wilcoxon vs FULL.
 *
 * Output:
 *   results/exp4a/group_ablation_raw.csv
 *   results/exp4a/group_ablation_summary.csv
 *
 * @author Le, Vo, Nguyen
 */
public class Exp4a_GroupAblation {

    private static final String OUT_DIR = "results/exp4a";
    private static final double TAU = 0.7;
    private static final int[] K_VALUES = {10};
    private static final int WARMUP = 2;
    private static final int REPS = 5;

    private static final String[] DATASETS = {
            "processed_data/chess_uncertain.txt"

    };

    /** Each ablation config: name -> PruningConfig generator. */
    private static final Map<String, java.util.function.Supplier<PruningConfig>> CONFIGS = new LinkedHashMap<>();
    static {
        CONFIGS.put("FULL",          PruningConfig::full);
        CONFIGS.put("NO_G1_frontier", () -> PruningConfig.full().withP1(false).withP2(false));
        CONFIGS.put("NO_G2_item",    () -> PruningConfig.full().withP3(false));
        CONFIGS.put("NO_G3_upbound", () -> PruningConfig.full().withP4(false).withP5(false));
        CONFIGS.put("NO_G4_tidset",  () -> PruningConfig.full().withP6(false).withP7(false));
        CONFIGS.put("ONLY_G1",       () -> PruningConfig.none().withP1(true).withP2(true));
        CONFIGS.put("ONLY_G2_G3_G4", () -> PruningConfig.full().withP1(false).withP2(false));
    }

    public static void main(String[] args) throws Exception {
        ExperimentRunner.ensureDir(OUT_DIR);

        for (String datasetPath : DATASETS) {
            System.out.println("====== " + datasetPath + " ======");
            UncertainDatabase db;
            try {
                db = UncertainDatabase.loadFromFile(datasetPath);//UncertaintyInjector.loadDefault(datasetPath);
            } catch (Exception e) {
                System.err.println("Could not load: " + e.getMessage());
                continue;
            }
            String datasetName = datasetPath;//db.getName();

            try (ExperimentRunner.CsvWriter raw = new ExperimentRunner.CsvWriter(
                    OUT_DIR + "/" + datasetName.replaceAll("[^a-zA-Z0-9_]", "_") + "_group_ablation_raw.csv", Arrays.asList(
                            "dataset", "k", "config", "rep",
                            "runtime_ms", "candidates_explored", "closure_checks"));
                 ExperimentRunner.CsvWriter sum = new ExperimentRunner.CsvWriter(
                         OUT_DIR + "/" + datasetName.replaceAll("[^a-zA-Z0-9_]", "_") + "_group_ablation_summary.csv", Arrays.asList(
                                 "dataset", "k", "config",
                                 "runtime_mean_ms", "runtime_std_ms",
                                 "speedup_vs_full", "p_value_vs_full",
                                 "closure_checks_mean"))) {

                for (int k : K_VALUES) {
                    System.out.println("---- k=" + k + " ----");

                    // Run FULL first (reference)
                    Map<String, double[]> runtimesByConfig = new LinkedHashMap<>();
                    Map<String, Long> closureSumByConfig = new LinkedHashMap<>();

                    for (String configName : CONFIGS.keySet()) {
                        System.out.println("  config: " + configName);
                        double[] runtimes = new double[REPS];
                        long closureSum = 0;

                        for (int rep = 0; rep < WARMUP + REPS; rep++) {
                            TUFCI_V1 m = new TUFCI_V1(db, TAU, k);
                            m.setPruningConfig(CONFIGS.get(configName).get());
                            final List<FrequentItemset>[] result = new List[]{ null };
                            ExperimentRunner.RunResult r = ExperimentRunner.timedRun(
                                    () -> result[0] = m.mine());

                            if (rep < WARMUP) continue;
                            int repIdx = rep - WARMUP;
                            runtimes[repIdx] = r.runtimeMs;
                            closureSum += m.getClosureChecks();

                            Map<String, Object> row = new LinkedHashMap<>();
                            row.put("dataset", datasetName);
                            row.put("k", k);
                            row.put("config", configName);
                            row.put("rep", repIdx);
                            row.put("runtime_ms", r.runtimeMs);
                            row.put("candidates_explored", m.getCandidatesExplored());
                            row.put("closure_checks", m.getClosureChecks());
                            raw.writeRow(row);
                        }

                        runtimesByConfig.put(configName, runtimes);
                        closureSumByConfig.put(configName, closureSum);
                    }

                    // Summary
                    double[] fullRuntimes = runtimesByConfig.get("FULL");
                    ExperimentRunner.Stats fullStats = ExperimentRunner.Stats.from(fullRuntimes);

                    for (Map.Entry<String, double[]> e : runtimesByConfig.entrySet()) {
                        ExperimentRunner.Stats s = ExperimentRunner.Stats.from(e.getValue());
                        double speedup = s.mean > 0 ? fullStats.mean / s.mean : 1.0;
                        double pVal = e.getKey().equals("FULL") ? 1.0 :
                                ExperimentRunner.wilcoxonSignedRank(fullRuntimes, e.getValue());
                        Map<String, Object> srow = new LinkedHashMap<>();
                        srow.put("dataset", datasetName);
                        srow.put("k", k);
                        srow.put("config", e.getKey());
                        srow.put("runtime_mean_ms", String.format("%.2f", s.mean));
                        srow.put("runtime_std_ms", String.format("%.2f", s.std));
                        srow.put("speedup_vs_full", String.format("%.3f", speedup));
                        srow.put("p_value_vs_full", String.format("%.6f", pVal));
                        srow.put("closure_checks_mean", closureSumByConfig.get(e.getKey()) / (double) REPS);
                        sum.writeRow(srow);
                    }
                }
            }
        }

        System.out.println("\n[Exp4a] Done. Results in " + OUT_DIR + "/");
    }
}