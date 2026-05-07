package experiment;

import domain.mining.PruningConfig;
import domain.mining.TUFCI_V1;
import domain.model.FrequentItemset;
import infrastructure.persistence.UncertainDatabase;

import java.util.*;

/**
 * Exp4c — Pairwise group synergy.
 *
 * Tests whether two groups together produce more benefit than the sum of their
 * individual contributions (positive synergy) or less (interference). For each
 * pair (Gi, Gj):
 *   marginal_i  = runtime(NO_Gi)  - runtime(FULL)
 *   marginal_j  = runtime(NO_Gj)  - runtime(FULL)
 *   joint_ij    = runtime(NO_Gi_Gj) - runtime(FULL)
 *   synergy_ij  = joint_ij - (marginal_i + marginal_j)
 *
 *   synergy_ij > 0  →  Gi and Gj are synergistic (joint loss > sum of losses)
 *   synergy_ij < 0  →  Gi and Gj are redundant   (joint loss < sum of losses)
 *   synergy_ij ≈ 0  →  Gi and Gj are independent
 *
 * This justifies WHY all four groups are kept (vs. eliminating redundant ones).
 *
 * Output:
 *   results/exp4c/group_synergy.csv
 *
 * @author Le, Vo, Nguyen
 */
public class Exp4c_GroupSynergy {

    private static final String OUT_DIR = "results/exp4c";
    private static final double TAU = 0.7;
    private static final int K = 10;
    private static final int WARMUP = 1;
    private static final int REPS = 3;

    private static final String[] DATASETS = {
            "processed_data/chess_uncertain.txt"

    };

    private static final String[] GROUPS = { "G1", "G2", "G3", "G4" };

    public static void main(String[] args) throws Exception {
        ExperimentRunner.ensureDir(OUT_DIR);

        for (String datasetPath : DATASETS) {
            System.out.println("====== " + datasetPath + " ======");
            UncertainDatabase db;
            try {
                db = UncertainDatabase.loadFromFile(datasetPath);//db = UncertainDatabase.loadFromFile(datasetPath);//db = UncertaintyInjector.loadDefault(datasetPath);
            } catch (Exception e) {
                System.err.println("Skipping: " + e.getMessage());
                continue;
            }
            String datasetName = db.getName();

            try (ExperimentRunner.CsvWriter w = new ExperimentRunner.CsvWriter(
                    OUT_DIR + "/" + datasetName + "_group_synergy.csv", Arrays.asList(
                            "dataset", "group_pair",
                            "runtime_full_ms",
                            "runtime_no_g1_ms", "runtime_no_g2_ms",
                            "runtime_no_both_ms",
                            "marginal_g1_ms", "marginal_g2_ms",
                            "joint_loss_ms", "synergy_ms",
                            "synergy_label"))) {

                double tFull = avgRuntime(db, K, PruningConfig::full);

                for (int i = 0; i < GROUPS.length; i++) {
                    for (int j = i + 1; j < GROUPS.length; j++) {
                        String gi = GROUPS[i];
                        String gj = GROUPS[j];
                        System.out.println("  pair: " + gi + " + " + gj);

                        double tNoI    = avgRuntime(db, K, () -> disable(gi));
                        double tNoJ    = avgRuntime(db, K, () -> disable(gj));
                        double tNoBoth = avgRuntime(db, K, () -> disable(gi, gj));

                        double marginalI = tNoI - tFull;
                        double marginalJ = tNoJ - tFull;
                        double joint     = tNoBoth - tFull;
                        double synergy   = joint - (marginalI + marginalJ);

                        String label = synergy > 5.0 ? "synergistic" :
                                       synergy < -5.0 ? "redundant" : "independent";

                        Map<String, Object> row = new LinkedHashMap<>();
                        row.put("dataset", datasetName);
                        row.put("group_pair", gi + "+" + gj);
                        row.put("runtime_full_ms", String.format("%.2f", tFull));
                        row.put("runtime_no_g1_ms", String.format("%.2f", tNoI));
                        row.put("runtime_no_g2_ms", String.format("%.2f", tNoJ));
                        row.put("runtime_no_both_ms", String.format("%.2f", tNoBoth));
                        row.put("marginal_g1_ms", String.format("%.2f", marginalI));
                        row.put("marginal_g2_ms", String.format("%.2f", marginalJ));
                        row.put("joint_loss_ms", String.format("%.2f", joint));
                        row.put("synergy_ms", String.format("%.2f", synergy));
                        row.put("synergy_label", label);
                        w.writeRow(row);
                    }
                }
            }
        }

        System.out.println("\n[Exp4c] Done. Results in " + OUT_DIR + "/");
    }

    private static PruningConfig disable(String... groups) {
        PruningConfig c = PruningConfig.full();
        for (String g : groups) {
            switch (g) {
                case "G1": c.withP1(false).withP2(false); break;
                case "G2": c.withP3(false); break;
                case "G3": c.withP4(false).withP5(false); break;
                case "G4": c.withP6(false).withP7(false); break;
            }
        }
        return c;
    }

    private static double avgRuntime(UncertainDatabase db, int k,
                                      java.util.function.Supplier<PruningConfig> cfg) {
        double[] runtimes = new double[REPS];
        for (int rep = 0; rep < WARMUP + REPS; rep++) {
            TUFCI_V1 m = new TUFCI_V1(db, TAU, k);
            m.setPruningConfig(cfg.get());
            final List<FrequentItemset>[] result = new List[]{ null };
            ExperimentRunner.RunResult r = ExperimentRunner.timedRun(() -> result[0] = m.mine());
            if (rep < WARMUP) continue;
            runtimes[rep - WARMUP] = r.runtimeMs;
        }
        return ExperimentRunner.Stats.from(runtimes).mean;
    }
}