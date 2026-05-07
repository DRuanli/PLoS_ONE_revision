package experiment;

import domain.mining.PruningConfig;
import domain.mining.TUFCI_V1;
import domain.model.FrequentItemset;
import infrastructure.persistence.UncertainDatabase;

import java.util.*;

/**
 * Exp4b — Group dominance heatmap.
 *
 * For each dataset, measure the speedup contributed by enabling ONLY that
 * group (vs. NONE) and the slowdown caused by removing that group (vs. FULL).
 * The diff between these two perspectives reveals each group's marginal
 * value on each dataset.
 *
 * The output supports a heatmap visualization showing, e.g., "G4 dominates
 * on Chess (dense), G1 dominates on Retail (sparse)" — directly addressing
 * the user's concern that no single pruning is universally dominant.
 *
 * Reviewer mapping:
 *   - Reviewer #1 #11: justify each pruning strategy's role
 *   - User concern: "some pruning dominates on certain datasets"
 *
 * Output:
 *   results/exp4b/group_dominance_heatmap.csv
 *
 * @author Le, Vo, Nguyen
 */
public class Exp4b_GroupDominance {

    private static final String OUT_DIR = "results/exp4b";
    private static final double TAU = 0.7;
    private static final int K = 10;
    private static final int WARMUP = 1;
    private static final int REPS = 3;

    private static final String[] DATASETS = {
            "processed_data/chess_uncertain.txt"
    };

    /** group name -> generator that turns ONLY that group on (others off). */
    private static final Map<String, java.util.function.Supplier<PruningConfig>> ONLY_GROUPS = new LinkedHashMap<>();
    /** group name -> generator that turns ONLY that group off (others on). */
    private static final Map<String, java.util.function.Supplier<PruningConfig>> NO_GROUPS = new LinkedHashMap<>();
    static {
        ONLY_GROUPS.put("G1", () -> PruningConfig.none().withP1(true).withP2(true));
        ONLY_GROUPS.put("G2", () -> PruningConfig.none().withP3(true));
        ONLY_GROUPS.put("G3", () -> PruningConfig.none().withP4(true).withP5(true));
        ONLY_GROUPS.put("G4", () -> PruningConfig.none().withP6(true).withP7(true));

        NO_GROUPS.put("G1", () -> PruningConfig.full().withP1(false).withP2(false));
        NO_GROUPS.put("G2", () -> PruningConfig.full().withP3(false));
        NO_GROUPS.put("G3", () -> PruningConfig.full().withP4(false).withP5(false));
        NO_GROUPS.put("G4", () -> PruningConfig.full().withP6(false).withP7(false));
    }

    public static void main(String[] args) throws Exception {
        ExperimentRunner.ensureDir(OUT_DIR);

        for (String datasetPath : DATASETS) {
            System.out.println("====== " + datasetPath + " ======");
            UncertainDatabase db;
            try {
                db = UncertainDatabase.loadFromFile(datasetPath);//UncertaintyInjector.loadDefault(datasetPath);
            } catch (Exception e) {
                System.err.println("Skipping: " + e.getMessage());
                continue;
            }
            String datasetName = db.getName();

            try (ExperimentRunner.CsvWriter w = new ExperimentRunner.CsvWriter(
                    OUT_DIR + "/" + datasetName + "_group_dominance_heatmap.csv", Arrays.asList(
                            "dataset", "group",
                            "runtime_full_ms", "runtime_none_ms",
                            "runtime_only_group_ms", "runtime_no_group_ms",
                            "marginal_benefit_pct", "exclusive_benefit_pct"))) {

                double tFull = avgRuntime(db, K, () -> PruningConfig.full());
                double tNone = avgRuntime(db, K, () -> PruningConfig.none());
                System.out.printf("  FULL=%.1fms  NONE=%.1fms%n", tFull, tNone);

                for (String group : Arrays.asList("G1", "G2", "G3", "G4")) {
                    double tOnly = avgRuntime(db, K, ONLY_GROUPS.get(group));
                    double tNo   = avgRuntime(db, K, NO_GROUPS.get(group));

                    // Marginal benefit: how much slower without this group, normalized
                    double marginal = tFull > 0 ? (tNo - tFull) / tFull * 100.0 : 0.0;
                    // Exclusive benefit: how much faster than NONE this group alone provides
                    double exclusive = tNone > 0 ? (tNone - tOnly) / tNone * 100.0 : 0.0;

                    Map<String, Object> row = new LinkedHashMap<>();
                    row.put("dataset", datasetName);
                    row.put("group", group);
                    row.put("runtime_full_ms", String.format("%.2f", tFull));
                    row.put("runtime_none_ms", String.format("%.2f", tNone));
                    row.put("runtime_only_group_ms", String.format("%.2f", tOnly));
                    row.put("runtime_no_group_ms", String.format("%.2f", tNo));
                    row.put("marginal_benefit_pct", String.format("%.2f", marginal));
                    row.put("exclusive_benefit_pct", String.format("%.2f", exclusive));
                    w.writeRow(row);
                    System.out.printf("    %s: only=%.1f  no=%.1f  marginal=%.1f%%  exclusive=%.1f%%%n",
                            group, tOnly, tNo, marginal, exclusive);
                }
            }
        }

        System.out.println("\n[Exp4b] Done. Results in " + OUT_DIR + "/");
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