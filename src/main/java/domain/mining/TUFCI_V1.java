package domain.mining;

import domain.support.SupportCalculator;
import infrastructure.persistence.UncertainDatabase;
import domain.model.*;
import domain.support.DirectConvolutionSupportCalculator;
import infrastructure.topK.TopKHeap;

import java.util.*;
import java.util.stream.Collectors;

/**
 * TUFCI_V1: Best-First Search + Full Pruning (P1–P7) + Descending Closure Order
 *
 * Primary algorithm variant for experiments.
 *
 * ═══════════════════════════════════════════════════════════════════════════════
 * CHANGES FOR MAJOR REVISION (PLoS ONE):
 *
 * 1. maxPqSize tracking
 *    The maximum size of the priority queue at any point during Phase 3 is now
 *    tracked and exposed via getMaxPqSize(). Required by Reviewer #4 for
 *    quantitative memory statistics.
 *
 * 2. P1 and P2 now respect pruningConfig (inherited from AbstractMiner)
 *    When pruningConfig.P1_earlyTermination = false, the Phase 2 early-
 *    termination break is skipped. Likewise for P2 in Phase 3.
 *    This enables Experiment 4 (ablation study) to isolate each pruning
 *    strategy's contribution without creating separate miner subclasses.
 *
 * 3. setPruningConfig() is defined in AbstractMiner; TUFCI_V1 uses the
 *    inherited pruningConfig field directly.
 * ═══════════════════════════════════════════════════════════════════════════════
 *
 * @author Dang Nguyen Le, Gia Huy Vo
 */
public class TUFCI_V1 extends AbstractMiner {

    // ════════════════════════════════════════════════════════════════════════
    // INSTANCE VARIABLES
    // ════════════════════════════════════════════════════════════════════════

    private PriorityQueue<FrequentItemset> pq;

    // ── Statistics for experiments ──
    private long candidatesExplored  = 0;
    private long candidatesPruned    = 0;
    private long supportCalculations = 0;
    private long closureChecks       = 0;

    /** Maximum PQ size observed during Phase 3. Required by Reviewer #4. */
    private int  maxPqSize           = 0;   // ← NEW

    private experiment.ThresholdTracker   thresholdTracker;
    private experiment.ConvergenceTracker convergenceTracker;

    // ════════════════════════════════════════════════════════════════════════
    // CONSTRUCTORS
    // ════════════════════════════════════════════════════════════════════════

    public TUFCI_V1(UncertainDatabase database, double tau, int k) {
        super(database, tau, k, new DirectConvolutionSupportCalculator(tau));
    }

    public TUFCI_V1(UncertainDatabase database, double tau, int k, SupportCalculator calculator) {
        super(database, tau, k, calculator);
    }

    // ════════════════════════════════════════════════════════════════════════
    // TRACKER SETTERS
    // ════════════════════════════════════════════════════════════════════════

    public void setThresholdTracker(experiment.ThresholdTracker tracker) {
        this.thresholdTracker = tracker;
    }

    public void setClosureMetrics(experiment.ClosureMetrics metrics) {
        this.closureMetrics = metrics;
    }

    public void setConvergenceTracker(experiment.ConvergenceTracker tracker) {
        this.convergenceTracker = tracker;
    }

    // ════════════════════════════════════════════════════════════════════════
    // PHASE 2 : Initialize Top-K with Closed Singletons
    //
    // P1 (Phase 2 Early Termination) is now guarded by pruningConfig.P1_earlyTermination.
    // ════════════════════════════════════════════════════════════════════════

    @Override
    protected void initializeTopKWithClosedSingletons(List<FrequentItemset> frequent1Itemsets) {
        this.topK    = new TopKHeap(getK());
        this.pq      = new PriorityQueue<>(FrequentItemset::compareForPriorityQueue);
        this.maxPqSize = 0;  // reset

        int minsup = 0;
        int processedItemCount = 0;
        long phase2StartTime = System.nanoTime();

        for (FrequentItemset fi : frequent1Itemsets) {
            int support = fi.getSupport();

            // ── P1: Phase 2 Early Termination (configurable for ablation) ─────
            if (pruningConfig.P1_earlyTermination && topK.isFull() && support < minsup) {
                break;
            }

            processedItemCount++;
            closureChecks++;

            boolean isClosed = checkClosure1Itemset(fi, support, frequent1Itemsets, minsup);

            if (isClosed) {
                boolean wasNotFull = !topK.isFull();
                boolean inserted   = topK.insert(fi);
                if (inserted && topK.isFull()) {
                    minsup = topK.getMinSupport();
                    if (convergenceTracker != null && wasNotFull) {
                        long elapsed = System.nanoTime() - phase2StartTime;
                        convergenceTracker.recordHeapFilled(processedItemCount, elapsed);
                    }
                }
            }
        }

        // Build frequent-items array (support >= minsup, descending order)
        List<Integer> frequentItemIndices = new ArrayList<>();
        for (int i = 0; i < processedItemCount; i++) {
            FrequentItemset fi = frequent1Itemsets.get(i);
            if (fi.getSupport() >= minsup) {
                frequentItemIndices.add(fi.getItems().get(0));
            }
        }

        this.frequentItemCount = frequentItemIndices.size();
        this.frequentItems     = new int[frequentItemCount];
        for (int i = 0; i < frequentItemCount; i++) {
            frequentItems[i] = frequentItemIndices.get(i);
        }

        // Seed PQ with 2-itemsets (computed as side-effect of Phase 2 closure checking)
        for (Map.Entry<Itemset, CachedFrequentItemset> entry : cache.entrySet()) {
            Itemset itemset = entry.getKey();
            if (itemset.size() == 2) {
                CachedFrequentItemset cached = entry.getValue();
                if (cached.getSupport() >= minsup) {
                    pq.add(cached.toFrequentItemset());
                }
            }
        }

        // Track initial PQ size
        if (!pq.isEmpty()) maxPqSize = pq.size();
    }

    // ════════════════════════════════════════════════════════════════════════
    // PHASE 3 : Best-First Mining
    //
    // P2 (Main Loop Threshold Pruning) is now guarded by pruningConfig.P2_thresholdPruning.
    // maxPqSize is updated after every PQ insertion.
    // ════════════════════════════════════════════════════════════════════════

    @Override
    protected void executePhase3(List<FrequentItemset> frequent1itemsets) {
        candidatesExplored = 0;
        candidatesPruned   = 0;
        long phase3StartTime = System.nanoTime();

        while (!pq.isEmpty()) {
            FrequentItemset candidate = pq.poll();
            candidatesExplored++;

            int threshold = getThreshold();

            // ── P2: Main Loop Threshold Pruning (configurable for ablation) ───
            // Best-First guarantees all remaining entries also have support ≤ candidate.
            // Safe to break (not just continue) when P2 is enabled.
            if (pruningConfig.P2_thresholdPruning && candidate.getSupport() < threshold) {
                candidatesPruned += pq.size() + 1;
                break;
            }

            closureChecks++;

            // P3–P7 are controlled via pruningConfig inside AbstractMiner.
            ClosureCheckResult result = checkClosureAndGenerateExtensions(candidate, threshold);

            if (result.isClosed()) {
                topK.insert(candidate);
            }

            if (convergenceTracker != null && candidatesExplored % 100 == 0) {
                long elapsed = System.nanoTime() - phase3StartTime;
                convergenceTracker.recordThresholdSnapshot(getThreshold(), candidatesExplored, elapsed);
            }

            int newThreshold = getThreshold();
            for (FrequentItemset ext : result.getExtensions()) {
                if (ext.getSupport() >= newThreshold) {
                    pq.add(ext);
                    // ── Track maximum PQ size (Reviewer #4 requirement) ────────
                    if (pq.size() > maxPqSize) maxPqSize = pq.size();
                } else {
                    candidatesPruned++;
                }
            }
        }

        if (convergenceTracker != null) {
            convergenceTracker.setFinalThreshold(getThreshold());
        }
    }

    // ════════════════════════════════════════════════════════════════════════
    // RESULTS
    // ════════════════════════════════════════════════════════════════════════

    @Override
    protected List<FrequentItemset> getTopKResults() {
        List<FrequentItemset> results = topK.getAll();
        results.sort(FrequentItemset::compareBySupport);
        return results;
    }

    // ════════════════════════════════════════════════════════════════════════
    // STATISTICS  (exposed for Experiment.java)
    // ════════════════════════════════════════════════════════════════════════

    public long getCandidatesExplored()   { return candidatesExplored;  }
    public long getCandidatesPruned()     { return candidatesPruned;    }
    public long getSupportCalculations()  { return supportCalculations; }
    public long getClosureChecks()        { return closureChecks;       }

    /**
     * Maximum size of the priority queue observed during Phase 3.
     * Required by Reviewer #4 for quantitative memory statistics.
     * Reported in results/exp1/main_comparison_raw.csv column 'max_queue_size'.
     */
    public int getMaxPqSize() { return maxPqSize; }

    public String getVariantName() { return "V1 (Best-First + Full Pruning)"; }
}