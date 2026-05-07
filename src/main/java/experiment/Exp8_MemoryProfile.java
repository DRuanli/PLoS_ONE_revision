package experiment;

import domain.mining.TUFCI_V1;
import domain.mining.TUFCI_V2;
import domain.model.FrequentItemset;
import experiment.baselines.ITUFP;
import experiment.baselines.TopKPFIM;
import infrastructure.persistence.UncertainDatabase;

import java.util.*;

/**
 * Exp8 — Memory profile.
 *
 * Reviewer mapping:
 *   - Reviewer #2 #4: provide quantitative statistics on priority queue size
 *     and peak memory usage
 *
 * For each dataset and algorithm, sample peak JVM heap usage during mining
 * and record the maximum search-frontier size (priority queue / stack /
 * over-mining buffer, depending on algorithm).
 *
 * Methodology:
 *   - Force GC before each run.
 *   - Sample heap usage every 50 ms during mining via a daemon thread.
 *   - Record maximum heap delta (rise above baseline).
 *   - Algorithm-specific structures: V1 uses PQ; V2 uses Stack;
 *     TopKPFIM and ITUFP use over-mining buffer / IMCUP-list.
 *
 * Output:
 *   results/exp8/memory_profile.csv
 *
 * @author Le, Vo, Nguyen
 */
public class Exp8_MemoryProfile {

    private static final String OUT_DIR = "results/exp8";
    private static final double TAU = 0.7;
    private static final int[] K_VALUES = {10};
    private static final int WARMUP = 1;
    private static final int REPS = 3;

    private static final String[] DATASETS = {
            "processed_data/chess_uncertain.txt"
    };

    private static final String[] ALGORITHMS = { "V1_BFS_Full", "V2_DFS_Full", "TopKPFIM", "ITUFP" };

    public static void main(String[] args) throws Exception {
        ExperimentRunner.ensureDir(OUT_DIR);

        for (String datasetPath : DATASETS) {
            System.out.println("====== " + datasetPath + " ======");
            UncertainDatabase db;
            try {
                db = UncertainDatabase.loadFromFile(datasetPath);
            } catch (Exception e) {
                System.err.println("Skipping: " + e.getMessage());
                continue;
            }
            String datasetName = db.getName();

            try (ExperimentRunner.CsvWriter w = new ExperimentRunner.CsvWriter(
                    OUT_DIR + "/" + datasetName + "_memory_profile.csv", Arrays.asList(
                            "dataset", "k", "algorithm", "rep",
                            "runtime_ms", "peak_heap_mb",
                            "max_frontier_size", "frontier_kind"))) {

                for (int k : K_VALUES) {
                    System.out.println("  k=" + k);
                    for (String algo : ALGORITHMS) {
                        System.out.println("    " + algo);

                        for (int rep = 0; rep < WARMUP + REPS; rep++) {
                            MemoryRun mr = profileOne(db, algo, k);
                            if (rep < WARMUP) continue;
                            int repIdx = rep - WARMUP;

                            Map<String, Object> row = new LinkedHashMap<>();
                            row.put("dataset", datasetName);
                            row.put("k", k);
                            row.put("algorithm", algo);
                            row.put("rep", repIdx);
                            row.put("runtime_ms", mr.runtimeMs);
                            row.put("peak_heap_mb", String.format("%.2f", mr.peakHeapBytes / 1024.0 / 1024.0));
                            row.put("max_frontier_size", mr.maxFrontierSize);
                            row.put("frontier_kind", mr.frontierKind);
                            w.writeRow(row);
                        }
                    }
                }
            }
        }

        System.out.println("\n[Exp8] Done. Results in " + OUT_DIR + "/");
    }

    private static class MemoryRun {
        long runtimeMs;
        long peakHeapBytes;
        long maxFrontierSize;
        String frontierKind;
    }

    /** Profile a single run with heap-sampling thread. */
    private static MemoryRun profileOne(UncertainDatabase db, String algo, int k) {
        Runtime rt = Runtime.getRuntime();
        System.gc();
        try { Thread.sleep(50); } catch (InterruptedException ignore) {}

        long heapBaseline = rt.totalMemory() - rt.freeMemory();
        final long[] peakHeap = { heapBaseline };
        final boolean[] running = { true };

        Thread sampler = new Thread(() -> {
            while (running[0]) {
                long h = rt.totalMemory() - rt.freeMemory();
                if (h > peakHeap[0]) peakHeap[0] = h;
                try { Thread.sleep(50); } catch (InterruptedException ignore) { return; }
            }
        });
        sampler.setDaemon(true);
        sampler.start();

        long startNs = System.nanoTime();

        MemoryRun result = new MemoryRun();
        switch (algo) {
            case "V1_BFS_Full": {
                TUFCI_V1 m = new TUFCI_V1(db, TAU, k);
                m.mine();
                result.maxFrontierSize = m.getMaxPqSize();
                result.frontierKind = "priority_queue";
                break;
            }
            case "V2_DFS_Full": {
                TUFCI_V2 m = new TUFCI_V2(db, TAU, k);
                m.mine();
                result.maxFrontierSize = m.getMaxStackSize();
                result.frontierKind = "stack";
                break;
            }
            case "TopKPFIM": {
                TopKPFIM m = new TopKPFIM(db, TAU, k);
                m.mine();
                result.maxFrontierSize = m.getMaxBufferSize();
                result.frontierKind = "over_mining_buffer";
                break;
            }
            case "ITUFP": {
                ITUFP m = new ITUFP(db, TAU, k);
                m.mine();
                result.maxFrontierSize = m.getMaxStackSize();
                result.frontierKind = "stack_plus_imcup";
                break;
            }
            default:
                throw new IllegalArgumentException("Unknown: " + algo);
        }

        long endNs = System.nanoTime();
        running[0] = false;
        try { sampler.join(200); } catch (InterruptedException ignore) {}

        result.runtimeMs = (endNs - startNs) / 1_000_000;
        result.peakHeapBytes = Math.max(0, peakHeap[0] - heapBaseline);
        return result;
    }
}