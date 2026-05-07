package domain.mining;

import domain.support.SupportCalculator;
import infrastructure.persistence.UncertainDatabase;
import infrastructure.persistence.Vocabulary;
import domain.model.*;
import domain.support.DirectConvolutionSupportCalculator;
import infrastructure.topK.TopKHeap;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

/**
 * TUFCI: Top-K Uncertain Frequent Closed Itemset Mining Algorithm
 *
 * <p>This class implements an efficient algorithm for mining the top-k frequent closed itemsets
 * from uncertain databases. An uncertain database contains transactions where each item has an
 * existential probability, making support calculation probabilistic rather than deterministic.</p>
 *
 * <p><b>Key Concepts:</b></p>
 * <ul>
 *   <li><b>Itemset:</b> A set of items (e.g., {bread, milk})</li>
 *   <li><b>Support:</b> The expected number of transactions containing an itemset</li>
 *   <li><b>Probability:</b> The likelihood that an itemset appears in at least one transaction</li>
 *   <li><b>Closed Itemset:</b> An itemset where no superset has the same support</li>
 *   <li><b>Top-K:</b> The k itemsets with highest support values</li>
 * </ul>
 *
 * <p><b>Algorithm Overview - Three Phases:</b></p>
 * <ol>
 *   <li><b>Phase 1 (computeAllSingletonSupports):</b> Compute probabilistic support for ALL singleton patterns (no filtering)</li>
 *   <li><b>Phase 2 (initializeTopKWithClosedSingletons):</b> Check closure for singletons, populate Top-K heap, seed priority queue</li>
 *   <li><b>Phase 3 (executePhase3):</b> Best-first search using priority queue to discover larger closed patterns</li>
 * </ol>
 *
 * <p><b>Pruning Strategies:</b> The algorithm uses multiple pruning techniques to improve efficiency:
 * early termination, upper bound filtering, subset-based pruning, and tidset-based pruning.</p>
 *
 * @author Dang Nguyen Le, Gia Huy Vo
 */
public class TUFCI extends AbstractMiner {

    // ==================== Instance Variables ====================

    /**
     * Priority queue for canonical mining in Phase 3.
     * Orders candidates by support (descending), then by itemset size (ascending),
     * then by probability (descending). This ordering ensures we explore the most
     * promising candidates first.
     */
    private PriorityQueue<FrequentItemset> pq;

    // ==================== Constructors ====================

    /**
     * Constructs a new TUFCI miner.
     *
     * @param database The uncertain database to mine
     * @param tau The probability threshold (not used in top-k mining but kept for compatibility)
     * @param k The number of top patterns to find
     */
    public TUFCI(UncertainDatabase database, double tau, int k) {
        super(database, tau, k, new DirectConvolutionSupportCalculator(tau));
    }

    /**
     * Constructor with custom support calculator.
     *
     * @param database uncertain database to mine
     * @param tau probability threshold
     * @param k number of top patterns
     * @param calculator custom support calculation strategy
     */
    public TUFCI(UncertainDatabase database, double tau, int k,
                                  SupportCalculator calculator) {
        super(database, tau, k, calculator);
    }

    // ==================== Phase 1: Compute Frequent 1-Itemsets ====================
    // Implemented in AbstractFrequentItemsetMiner - no override needed

    // ==================== Phase 2: Initialize Data Structures ====================

    /**
     * Phase 2: Initializes Top-K heap, performs closure checking on 1-itemsets,
     * and prepares the priority queue for Phase 3.
     *
     * <p><b>Algorithm Steps:</b></p>
     * <ol>
     *   <li>Initialize Top-K heap to store the best k patterns</li>
     *   <li>Initialize priority queue for canonical mining</li>
     *   <li>For each 1-itemset (in support-descending order):
     *     <ul>
     *       <li>Apply early termination pruning if Top-K is full and support < minsup</li>
     *       <li>Check closure: verify no other single item has same support when combined</li>
     *       <li>If closed, insert into Top-K heap</li>
     *       <li>Update minsup threshold if Top-K becomes full</li>
     *     </ul>
     *   </li>
     *   <li>Build list of frequent items (support >= minsup)</li>
     *   <li>Add all 2-itemsets with support >= minsup to priority queue</li>
     * </ol>
     *
     * <p><b>Closure Check for 1-itemsets:</b></p>
     * <p>A single item A is closed if for all other items B:
     * support(A ∪ B) < support(A). This means no superset has the same support.</p>
     *
     * <p><b>Why Check 2-itemsets?</b> During closure checking, we compute support for
     * all 2-itemsets. We cache these and add promising ones to the priority queue
     * to start Phase 3.</p>
     *
     * @param frequent1Itemsets The sorted list of 1-itemsets from Phase 1
     */
    @Override
    protected void initializeTopKWithClosedSingletons(List<FrequentItemset> frequent1Itemsets) {
        // Initialize Top-K heap to track the k best patterns
        this.topK = new TopKHeap(getK());

        /**
         * Initialize priority queue with custom comparator using static method.
         * Ordering: support (DESC) → size (ASC) → probability (DESC)
         */
        this.pq = new PriorityQueue<>(FrequentItemset::compareForPriorityQueue);

        // Minimum support threshold - starts at 0, increases as Top-K fills up
        int minsup = 0;

        // Track how many items we process before early termination
        int processedItemCount = 0;

        // Process each 1-itemset in support-descending order
        for (FrequentItemset fi : frequent1Itemsets) {
            int support = fi.getSupport();
            double probability = fi.getProbability();

            /**
             * Pruning Strategy 1: Phase 1 Early Termination
             *
             * If Top-K is full and current support < minsup, we can stop because:
             * 1. All remaining 1-itemsets have even lower support (sorted order)
             * 2. By anti-monotonicity, their supersets also have lower support
             * 3. Therefore, they cannot enter the Top-K
             */
            if (topK.isFull() && support < minsup) {
                break;
            }

            processedItemCount++;

            // Check if this 1-itemset is closed
            boolean isClosed = checkClosure1Itemset(fi, support, frequent1Itemsets, minsup);

            // If closed, try to insert into Top-K
            if (isClosed) {
                boolean inserted = topK.insert(fi);

                // Update minimum support threshold when Top-K becomes full
                if (inserted && topK.isFull()) {
                    minsup = topK.getMinSupport();
                }
            }
        }

        /**
         * Build the list of frequent items (those meeting minsup threshold).
         * These are the only items we'll consider for extension in Phase 3.
         */
        List<Integer> frequentItemIndices = new ArrayList<>();

        for (int i = 0; i < processedItemCount; i++) {
            FrequentItemset fi = frequent1Itemsets.get(i);

            if (fi.getSupport() >= minsup) {
                // Extract the item ID from the singleton itemset
                frequentItemIndices.add(fi.getItems().get(0));
            }
        }

        // Store frequent items as array for fast iteration in Phase 3
        this.frequentItemCount = frequentItemIndices.size();
        this.frequentItems = new int[frequentItemCount];
        for (int i = 0; i < frequentItemCount; i++) {
            frequentItems[i] = frequentItemIndices.get(i);
        }

        /**
         * Seed the priority queue with 2-itemsets.
         *
         * During closure checking, we already computed support for all 2-itemsets.
         * Add those with support >= minsup to the priority queue to start Phase 3.
         * This bootstraps the canonical mining process.
         */
        for (Map.Entry<Itemset, CachedFrequentItemset> entry : cache.entrySet()) {
            Itemset itemset = entry.getKey();

            if (itemset.size() == 2) {
                CachedFrequentItemset cached = entry.getValue();

                if (cached.getSupport() >= minsup) {
                    // Add FrequentItemset to PQ (tidset not needed in PQ)
                    pq.add(cached.toFrequentItemset());
                }
            }
        }
    }

    // ==================== Phase 3: Recursive Mining ====================

    /**
     * Phase 3: Performs canonical mining to discover all top-k closed itemsets.
     *
     * <p><b>Algorithm Overview:</b></p>
     * <p>Uses a best-first search strategy with a priority queue. For each candidate pattern X:
     * <ol>
     *   <li>Check if X can potentially enter Top-K (threshold pruning)</li>
     *   <li>Check if X is closed (no superset has the same support)</li>
     *   <li>If closed, insert into Top-K heap</li>
     *   <li>Generate extensions X ∪ {item} for all valid items</li>
     *   <li>Add promising extensions to priority queue</li>
     * </ol>
     * </p>
     *
     * <p><b>Canonical Order:</b></p>
     * <p>Extensions are generated in canonical order: only add items with ID greater than
     * the maximum item in X. This ensures each pattern is generated exactly once,
     * avoiding redundant computation.</p>
     *
     * <p><b>Dynamic Threshold:</b></p>
     * <p>As better patterns are found and enter Top-K, the minimum support threshold
     * increases, allowing more aggressive pruning of unpromising candidates.</p>
     *
     * @param frequent1itemsets The list of 1-itemsets (not used in this implementation)
     */
    @Override
    protected void executePhase3(List<FrequentItemset> frequent1itemsets) {
        // Process candidates in priority order until queue is empty
        while (!pq.isEmpty()) {
            // Get the most promising candidate (highest support)
            FrequentItemset candidate = pq.poll();

            int threshold = getThreshold();

            /**
             * Pruning Strategy 2: Main Loop Threshold Pruning
             *
             * If candidate support < current threshold, skip it because:
             * 1. It cannot enter the Top-K (not good enough)
             * 2. All remaining candidates in PQ also have support <= this candidate (PQ ordering)
             * 3. Therefore, we can terminate the entire search
             */
            if (candidate.getSupport() < threshold) {
                break;
            }

            /**
             * Check closure and generate extensions.
             *
             * This method:
             * 1. Verifies if the candidate is closed
             * 2. Generates all valid extensions in canonical order
             * 3. Applies multiple pruning strategies to reduce computation
             *
             * Pass cached threshold to avoid redundant synchronized getThreshold() call
             */
            ClosureCheckResult result = checkClosureAndGenerateExtensions(candidate, threshold);

            // If closed, add to Top-K (which will update threshold if needed)
            if (result.isClosed()) {
                topK.insert(candidate);
            }

            /**
             * Add extensions to priority queue for future exploration.
             *
             * Get updated threshold (may have changed after insertion to Top-K)
             * and only add extensions that meet the threshold.
             */
            int newThreshold = getThreshold();
            for (FrequentItemset ext : result.getExtensions()) {
                if (ext.getSupport() >= newThreshold) {
                    pq.add(ext);
                }
            }
        }
    }

    // ==================== Result Retrieval ====================

    /**
     * Retrieves the final top-k patterns from the heap.
     *
     * <p>Extracts all patterns from the Top-K heap and sorts them by:
     * <ol>
     *   <li>Support (descending) - highest support first</li>
     *   <li>Probability (descending) - break ties with probability</li>
     * </ol>
     * </p>
     *
     * @return List of top-k patterns sorted by support then probability
     */
    @Override
    protected List<FrequentItemset> getTopKResults() {
        // Get all patterns from heap
        List<FrequentItemset> results = topK.getAll();

        // Sort by support DESC, then probability DESC using static comparator
        results.sort(FrequentItemset::compareBySupport);

        return results;
    }

    // ==================== Utility Methods & Closure Checking ====================
    // All utility methods and closure checking methods are now implemented in
    // AbstractFrequentItemsetMiner to eliminate code duplication.
    // Available methods: createSingletonItemset, getThreshold, getItemSupport,
    // getMaxItemIndex, checkClosure1Itemset, checkClosureAndGenerateExtensions
}
