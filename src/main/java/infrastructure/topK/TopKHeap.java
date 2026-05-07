package infrastructure.topK;

import domain.model.Itemset;
import domain.model.FrequentItemset;

import java.util.*;

/**
 * TopKHeap - Min-heap data structure for maintaining top-K patterns by support.
 *
 * ═══════════════════════════════════════════════════════════════════════════════
 * PURPOSE
 * ═══════════════════════════════════════════════════════════════════════════════
 *
 * During mining, we need to track the K patterns with highest support.
 * New patterns are discovered continuously, and we need to:
 *   1. Quickly check if a new pattern can enter top-K
 *   2. Efficiently replace the minimum if new pattern is better
 *   3. Provide dynamic threshold for pruning
 *
 * Solution: MIN-HEAP of size K
 *   - Root = pattern with MINIMUM support in top-K
 *   - New pattern enters only if support > root.support
 *   - Threshold = root.support (for pruning during mining)
 *
 * ═══════════════════════════════════════════════════════════════════════════════
 * WHY MIN-HEAP (not MAX-HEAP)?
 * ═══════════════════════════════════════════════════════════════════════════════
 *
 * We want TOP-K (highest support), so why use MIN-heap?
 *
 * Key insight: We only care about the MINIMUM of the top-K!
 *   - To check if new pattern can enter: compare with minimum
 *   - To replace: remove minimum, add new pattern
 *   - Threshold for pruning: the minimum support
 *
 * MIN-heap gives O(1) access to minimum!
 *
 * MAX-heap would require O(K) to find minimum for threshold.
 *
 * ═══════════════════════════════════════════════════════════════════════════════
 * OPERATIONS
 * ═══════════════════════════════════════════════════════════════════════════════
 *
 * insert(itemset, support, probability):
 *   - If heap not full: add directly, O(log K)
 *   - If heap full and support > min: replace min, O(log K)
 *   - If heap full and support ≤ min: reject, O(1)
 *
 * getMinSupport():
 *   - Return threshold for pruning
 *   - O(1) - just peek at root
 *
 * isFull():
 *   - Check if heap has K patterns
 *   - O(1)
 *
 * getAll():
 *   - Return all patterns (for final results)
 *   - O(K)
 *
 * @author Dang Nguyen Le, Gia Huy Vo
 */
public class TopKHeap {

    /**
     * Maximum number of patterns to maintain (the K in top-K).
     */
    private final int k;

    /**
     * Min-heap ordered by support (ascending), then probability (ascending).
     * Root is always the pattern with minimum support.
     *
     * Comparator: (a, b) -> compare(a.getSupport(), b.getSupport())
     *   - Returns negative if a.getSupport() < b.getSupport() (a comes first)
     *   - This makes it a MIN-heap (smallest at root)
     */
    private final PriorityQueue<FrequentItemset> heap;

    /**
     * Set of itemsets already in heap (for deduplication).
     * Prevents inserting same itemset multiple times.
     */
    private final Set<Itemset> seenItemsets;

    /**
     * Constructor.
     *
     * @param k number of top patterns to maintain
     */
    public TopKHeap(int k) {
        this.k = k;

        // MIN-heap: smallest support at root
        // Comparator returns negative when a < b, making a come first (root)
        this.heap = new PriorityQueue<>((a, b) -> {
            // Primary: compare by support (ascending for min-heap)
            int cmp = Integer.compare(a.getSupport(), b.getSupport());

            // Secondary: compare by probability (ascending)
            // If supports equal, lower probability = more likely to be replaced
            if (cmp == 0) {
                cmp = Double.compare(a.getProbability(), b.getProbability());
            }

            return cmp;
        });

        // Track seen itemsets for deduplication
        this.seenItemsets = new HashSet<>();
    }

    /**
     * Insert a FrequentItemset into the heap.
     *
     * ═══════════════════════════════════════════════════════════════════════
     * ALGORITHM:
     * ═══════════════════════════════════════════════════════════════════════
     *
     * 1. DEDUPLICATION CHECK:
     *    - If itemset already seen, reject (return false)
     *    - Same itemset may be generated via different extension paths
     *
     * 2. HEAP NOT FULL (size < k):
     *    - Add pattern directly
     *    - No comparison needed
     *
     * 3. HEAP FULL (size = k):
     *    - Compare with minimum (root)
     *    - If new pattern is STRICTLY BETTER: replace minimum
     *    - Otherwise: reject
     *
     * "Strictly better" means:
     *   - Higher support, OR
     *   - Same support but higher probability
     *
     * ═══════════════════════════════════════════════════════════════════════
     *
     * Thread-safe: synchronized for concurrent access from mining.
     *
     * @param fi FrequentItemset to insert
     * @return true if inserted, false if duplicate or not competitive
     */
    public synchronized boolean insert(FrequentItemset fi) {
        // ─────────────────────────────────────────────────────────────
        // STEP 1: Deduplication check
        // Mining may discover same itemset via different extension paths
        // ─────────────────────────────────────────────────────────────
        if (seenItemsets.contains(fi)) {
            return false;  // Already in heap
        }

        // ─────────────────────────────────────────────────────────────
        // STEP 2: Heap not full - add directly
        // ─────────────────────────────────────────────────────────────
        if (heap.size() < k) {
            // Add FrequentItemset to heap
            heap.offer(fi);

            // Track for deduplication
            seenItemsets.add(fi);

            return true;
        }

        // ─────────────────────────────────────────────────────────────
        // STEP 3: Heap full - compare with minimum
        // ─────────────────────────────────────────────────────────────
        FrequentItemset min = heap.peek();  // O(1) - root of min-heap

        // Check if new pattern is STRICTLY better than minimum
        // Strict inequality ensures we only replace when truly better
        boolean isBetter = (fi.getSupport() > min.getSupport()) ||
                          (fi.getSupport() == min.getSupport() && fi.getProbability() > min.getProbability());

        if (isBetter) {
            // Remove minimum from heap
            heap.poll();  // O(log K)

            // Remove from seen set (allow re-insertion if rediscovered)
            seenItemsets.remove(min);

            // Add new pattern
            heap.offer(fi);  // O(log K)
            seenItemsets.add(fi);

            return true;
        }

        // New pattern not competitive enough
        return false;
    }

    /**
     * Check if heap is full (contains k patterns).
     *
     * When full:
     *   - getMinSupport() returns valid threshold
     *   - New patterns must beat minimum to enter
     *
     * When not full:
     *   - Still collecting initial K patterns
     *   - Any pattern can enter (no threshold yet)
     *
     * Thread-safe: synchronized for concurrent access.
     *
     * @return true if heap contains k patterns, false otherwise
     */
    public synchronized boolean isFull() {
        return heap.size() >= k;
    }

    /**
     * Get minimum support threshold for pruning.
     *
     * ═══════════════════════════════════════════════════════════════════════
     * CRITICAL: RETURN 0 WHEN NOT FULL
     * ═══════════════════════════════════════════════════════════════════════
     *
     * When heap is not full (size < k), we haven't found K patterns yet.
     * Returning actual minimum would incorrectly prune candidates that
     * could have made it into top-K.
     *
     * Example:
     *   k = 10, heap has 3 patterns with supports [100, 90, 80]
     *   If we return min = 80, candidate with support 75 would be pruned
     *   But support 75 could be 4th best overall!
     *
     * Solution: Return 0 when not full → no threshold-based pruning
     *           Mining algorithm accepts all patterns until TopK fills
     *
     * ═══════════════════════════════════════════════════════════════════════
     *
     * Thread-safe: synchronized for concurrent access.
     *
     * @return minimum support in heap if full, 0 otherwise
     */
    public synchronized int getMinSupport() {
        // CRITICAL: Don't return threshold until we have K patterns
        if (!isFull()) {
            return 0;
        }

        // Return minimum (root of min-heap)
        return heap.peek().getSupport();
    }

    /**
     * Get all patterns currently in the heap.
     *
     * Returns copy of heap contents (not the heap itself).
     * Used for final results after mining completes.
     *
     * Note: Patterns are NOT sorted. Caller should sort if needed using:
     *   patterns.sort(FrequentItemset::compareBySupport);
     *
     * Thread-safe: synchronized for concurrent access.
     *
     * @return list of all FrequentItemsets in heap
     */
    public synchronized List<FrequentItemset> getAll() {
        // Return copy to prevent external modification of heap
        return new ArrayList<>(heap);
    }
}