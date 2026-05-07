package domain.mining;

import domain.support.SupportCalculator;
import infrastructure.persistence.UncertainDatabase;
import infrastructure.persistence.Vocabulary;
import domain.model.*;
import domain.support.DirectConvolutionSupportCalculator;
import infrastructure.topK.TopKHeap;

import java.util.*;
import java.util.stream.Collectors;

/**
 * TUFCI_V2: Depth-First Search + Full Pruning (P1-P7) + Enumeration Order
 *
 * DFS variant for comparison with Best-First Search.
 *
 * <h2>Algorithm Characteristics:</h2>
 * <ul>
 *   <li><b>Search Strategy:</b> DFS (Stack - LIFO order)</li>
 *   <li><b>Pruning:</b> Full (P1-P7 all enabled)</li>
 *   <li><b>Closure Order:</b> Enumeration (natural item order)</li>
 *   <li><b>Early Termination:</b> No (cannot break early due to mixed support in stack)</li>
 * </ul>
 *
 * <h2>Key Differences from V1:</h2>
 * <pre>
 * ┌─────────────────┬──────────────────────┬──────────────────────┐
 * │ Aspect          │ V1 (Best-First)      │ V2 (DFS)             │
 * ├─────────────────┼──────────────────────┼──────────────────────┤
 * │ Data Structure  │ PriorityQueue        │ Stack (Deque)        │
 * │ Selection       │ Highest support      │ Most recent (LIFO)   │
 * │ Early Term      │ Yes (break)          │ No (continue)        │
 * │ Threshold Rise  │ Fast                 │ Slower               │
 * │ Memory          │ O(candidates)        │ O(depth × branch)    │
 * └─────────────────┴──────────────────────┴──────────────────────┘
 * </pre>
 *
 * @author Dang Nguyen Le, Gia Huy Vo
 */
public class TUFCI_V2 extends AbstractMiner {

    // ==================== Instance Variables ====================

    private Deque<FrequentItemset> stack;

    // Statistics for experiments
    private long candidatesExplored = 0;
    private long candidatesPruned = 0;
    private long supportCalculations = 0;
    private long closureChecks = 0;
    private long maxStackSize = 0;
    private experiment.ThresholdTracker thresholdTracker;
    private experiment.ConvergenceTracker convergenceTracker;

    // ==================== Constructors ====================

    public TUFCI_V2(UncertainDatabase database, double tau, int k) {
        super(database, tau, k, new DirectConvolutionSupportCalculator(tau));
    }

    public TUFCI_V2(UncertainDatabase database, double tau, int k, SupportCalculator calculator) {
        super(database, tau, k, calculator);
    }

    public void setThresholdTracker(experiment.ThresholdTracker tracker) {
        this.thresholdTracker = tracker;
    }

    public void setClosureMetrics(experiment.ClosureMetrics metrics) {
        this.closureMetrics = metrics;
    }

    public void setConvergenceTracker(experiment.ConvergenceTracker tracker) {
        this.convergenceTracker = tracker;
    }

    // ==================== Phase 2: Initialize ====================

    @Override
    protected void initializeTopKWithClosedSingletons(List<FrequentItemset> frequent1Itemsets) {
        this.topK = new TopKHeap(getK());
        this.stack = new ArrayDeque<>();

        int minsup = 0;
        int processedItemCount = 0;
        long phase2StartTime = System.nanoTime();

        // Process 1-itemsets with P1: Early Termination
        for (FrequentItemset fi : frequent1Itemsets) {
            int support = fi.getSupport();

            // P1a: Phase 2 Early Termination
            if (topK.isFull() && support < minsup) {
                break;
            }

            processedItemCount++;
            closureChecks++;

            boolean isClosed = checkClosure1Itemset(fi, support, frequent1Itemsets, minsup);

            if (isClosed) {
                boolean wasNotFull = !topK.isFull();
                boolean inserted = topK.insert(fi);
                if (inserted && topK.isFull()) {
                    minsup = topK.getMinSupport();
                    // Record when heap becomes full
                    if (convergenceTracker != null && wasNotFull) {
                        long elapsed = System.nanoTime() - phase2StartTime;
                        convergenceTracker.recordHeapFilled(processedItemCount, elapsed);
                    }
                }
            }
        }

        // Build frequent items array
        List<Integer> frequentItemIndices = new ArrayList<>();
        for (int i = 0; i < processedItemCount; i++) {
            FrequentItemset fi = frequent1Itemsets.get(i);
            if (fi.getSupport() >= minsup) {
                frequentItemIndices.add(fi.getItems().get(0));
            }
        }

        this.frequentItemCount = frequentItemIndices.size();
        this.frequentItems = new int[frequentItemCount];
        for (int i = 0; i < frequentItemCount; i++) {
            frequentItems[i] = frequentItemIndices.get(i);
        }

        // Seed stack with 2-itemsets (reverse order so high support on top)
        List<FrequentItemset> twoItemsets = new ArrayList<>();
        for (Map.Entry<Itemset, CachedFrequentItemset> entry : cache.entrySet()) {
            Itemset itemset = entry.getKey();
            if (itemset.size() == 2) {
                CachedFrequentItemset cached = entry.getValue();
                if (cached.getSupport() >= minsup) {
                    twoItemsets.add(cached.toFrequentItemset());
                }
            }
        }

        // Sort ascending so higher support ends up on top after push
        twoItemsets.sort((a, b) -> Integer.compare(a.getSupport(), b.getSupport()));
        for (FrequentItemset fi : twoItemsets) {
            stack.push(fi);
        }

        maxStackSize = stack.size();
    }

    // ==================== Phase 3: DFS Mining ====================

    @Override
    protected void executePhase3(List<FrequentItemset> frequent1itemsets) {
        candidatesExplored = 0;
        candidatesPruned = 0;
        long phase3StartTime = System.nanoTime();

        while (!stack.isEmpty()) {
            // Track max stack size
            if (stack.size() > maxStackSize) {
                maxStackSize = stack.size();
            }

            FrequentItemset candidate = stack.pop();
            candidatesExplored++;

            int threshold = getThreshold();

            // ★ KEY [Search]: DFS — P2 continue = skip one only
            // DFS cannot early terminate (mixed support in stack)
            // Must use CONTINUE, not BREAK
            if (candidate.getSupport() < threshold) {
                candidatesPruned++;
                continue;  // Skip but don't break
            }

            closureChecks++;
            // ★ KEY [Pruning]: Full — P4/P5/P6/P7 shortcuts active
            ClosureCheckResult result = checkClosureAndGenerateExtensions(candidate, threshold);

            if (result.isClosed()) {
                topK.insert(candidate);
            }

            // Record convergence snapshot every 100 candidates
            if (convergenceTracker != null && candidatesExplored % 100 == 0) {
                long elapsed = System.nanoTime() - phase3StartTime;
                convergenceTracker.recordThresholdSnapshot(getThreshold(), candidatesExplored, elapsed);
            }

            // Push extensions (reverse order for DFS behavior)
            int newThreshold = getThreshold();
            List<FrequentItemset> validExtensions = new ArrayList<>();

            for (FrequentItemset ext : result.getExtensions()) {
                if (ext.getSupport() >= newThreshold) {
                    validExtensions.add(ext);
                } else {
                    candidatesPruned++;
                }
            }

            // Sort ascending so higher support on top
            validExtensions.sort((a, b) -> Integer.compare(a.getSupport(), b.getSupport()));
            for (FrequentItemset ext : validExtensions) {
                stack.push(ext);
            }
        }

        // Set final threshold for convergence calculations
        if (convergenceTracker != null) {
            convergenceTracker.setFinalThreshold(getThreshold());
        }
    }

    // ==================== Results ====================

    @Override
    protected List<FrequentItemset> getTopKResults() {
        List<FrequentItemset> results = topK.getAll();
        results.sort(FrequentItemset::compareBySupport);
        return results;
    }

    // ==================== Statistics ====================

    public long getCandidatesExplored() { return candidatesExplored; }
    public long getCandidatesPruned() { return candidatesPruned; }
    public long getSupportCalculations() { return supportCalculations; }
    public long getClosureChecks() { return closureChecks; }
    public long getMaxStackSize() { return maxStackSize; }

    public String getVariantName() { return "V2 (DFS + Full Pruning)"; }
}