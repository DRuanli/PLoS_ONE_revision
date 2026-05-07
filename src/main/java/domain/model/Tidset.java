package domain.model;

import shared.Constants;

import java.util.ArrayList;
import java.util.List;

/**
 * Tidset - Transaction ID Set with probabilities (Vertical Database Structure).
 *
 * In vertical database representation, instead of storing:
 *   Horizontal: Transaction -> {items}
 *   T1: {A, B, C}
 *   T2: {A, C}
 *
 * We store:
 *   Vertical: Item -> {transactions containing item}
 *   A: {T1, T2}
 *   B: {T1}
 *   C: {T1, T2}
 *
 * For uncertain databases, each entry also stores probability:
 *   A: {(T1, 0.9), (T2, 0.7)} means:
 *     - A appears in T1 with 90% probability
 *     - A appears in T2 with 70% probability
 *
 * Key Operation: Intersection for computing itemset support
 *   tidset({A}) ∩ tidset({B}) = tidset({A,B})
 *
 * @author Dang Nguyen Le, Gia Huy Vo
 */
public class Tidset {

    /**
     * List of (transaction ID, probability) pairs.
     * Sorted by transaction ID for efficient merge-join intersection.
     *
     * Example: [(0, 0.9), (2, 0.7), (5, 0.8)]
     *   - Transaction 0 contains item with prob 0.9
     *   - Transaction 2 contains item with prob 0.7
     *   - Transaction 5 contains item with prob 0.8
     */
    private final List<TIDProb> entries;

    /**
     * Default constructor - creates empty tidset.
     */
    public Tidset() {
        this.entries = new ArrayList<>();
    }

    /**
     * Constructor from existing entries list.
     * Creates defensive copy to ensure immutability.
     *
     * @param entries list of TIDProb entries to copy
     */
    public Tidset(List<TIDProb> entries) {
        // Defensive copy - changes to original list don't affect this tidset
        this.entries = new ArrayList<>(entries);
    }

    /**
     * Add a transaction ID with its probability.
     *
     * @param tid  transaction ID (must be non-negative)
     * @param prob probability of item in this transaction (must be in [0,1])
     * @throws IllegalArgumentException if tid < 0 or prob invalid
     */
    public void add(int tid, double prob) {
        // Validate transaction ID
        if (tid < 0) {
            throw new IllegalArgumentException(
                "Transaction ID must be non-negative, got: " + tid
            );
        }

        // Validate probability is a valid number
        if (Double.isNaN(prob) || Double.isInfinite(prob)) {
            throw new IllegalArgumentException(
                "Probability cannot be NaN or Infinite"
            );
        }

        // Validate probability is in valid range
        if (prob < 0.0 || prob > 1.0) {
            throw new IllegalArgumentException(
                "Probability must be in [0.0, 1.0], got: " + prob
            );
        }

        // Add new entry
        entries.add(new TIDProb(tid, prob));
    }

    /**
     * Get number of transactions in this tidset.
     *
     * @return count of entries
     */
    public int size() {
        return entries.size();
    }

    /**
     * Check if tidset is empty.
     *
     * @return true if no transactions, false otherwise
     */
    public boolean isEmpty() {
        return entries.isEmpty();
    }

    /**
     * Get all entries (for iteration).
     *
     * @return list of TIDProb entries
     */
    public List<TIDProb> getEntries() {
        return entries;
    }

    /**
     * Intersect this tidset with another using merge-join algorithm.
     *
     * Algorithm: Two-pointer technique on sorted lists
     *   - Both lists must be sorted by tid (ascending)
     *   - If tids match: combine probabilities and add to result
     *   - Otherwise: advance pointer with smaller tid
     *
     * Probability combination (Independence Assumption):
     *   P(A AND B) = P(A) × P(B)
     *
     * Time Complexity: O(|this| + |other|) - linear scan
     * Space Complexity: O(min(|this|, |other|)) - result size
     *
     * Example:
     *   tidset(A) = [(1, 0.8), (2, 0.9), (4, 0.7)]
     *   tidset(B) = [(2, 0.6), (3, 0.5), (4, 0.8)]
     *   Result    = [(2, 0.54), (4, 0.56)]
     *     - T2: 0.9 × 0.6 = 0.54
     *     - T4: 0.7 × 0.8 = 0.56
     *
     * @param other tidset to intersect with
     * @return new tidset containing common transactions with combined probabilities
     */
    public Tidset intersect(Tidset other) {
        // Pre-allocate with estimated size to avoid ArrayList resizing
        int estimatedSize = Math.min(this.entries.size(), other.entries.size());
        List<TIDProb> resultEntries = new ArrayList<>(estimatedSize);

        // Two pointers for merge-join
        int i = 0, j = 0;
        List<TIDProb> list1 = this.entries;
        List<TIDProb> list2 = other.entries;

        // Merge-join: scan both sorted lists simultaneously
        while (i < list1.size() && j < list2.size()) {
            TIDProb e1 = list1.get(i);
            TIDProb e2 = list2.get(j);

            if (e1.tid == e2.tid) {
                // Match found: combine probabilities assuming independence
                // P(A ∩ B in transaction t) = P(A in t) × P(B in t)
                double combinedProb = e1.prob * e2.prob;

                // Underflow protection: prevent probability from becoming
                // too small (causes numerical instability in GF computation)
                if (combinedProb < Constants.MIN_PROB) {
                    combinedProb = Constants.MIN_PROB;
                }

                resultEntries.add(new TIDProb(e1.tid, combinedProb));

                // Advance both pointers
                i++;
                j++;
            } else if (e1.tid < e2.tid) {
                // e1.tid not in list2, advance i
                i++;
            } else {
                // e2.tid not in list1, advance j
                j++;
            }
        }
        // Remaining elements in either list have no match, ignored

        return new Tidset(resultEntries);
    }

    /**
     * Convert sparse tidset to dense probability array.
     *
     * Used by GF/FFT support calculators which need array format.
     *
     * Example:
     *   Tidset: [(1, 0.8), (3, 0.6)]
     *   totalTransactions: 5
     *   Result: [0.0, 0.8, 0.0, 0.6, 0.0]
     *           index: 0    1    2    3    4
     *
     * @param totalTransactions total number of transactions in database
     * @return probability array indexed by transaction ID
     */
    public double[] toTransactionProbabilities(int totalTransactions) {
        // Initialize all probabilities to 0 (item not present)
        double[] probs = new double[totalTransactions];

        // Fill in probabilities for transactions that contain item
        for (TIDProb entry : entries) {
            // Bounds check (defensive programming)
            if (entry.tid < totalTransactions) {
                probs[entry.tid] = entry.prob;
            }
        }

        return probs;
    }

    @Override
    public String toString() {
        if (entries.isEmpty()) {
            return "[]";
        }

        StringBuilder sb = new StringBuilder();
        sb.append("[");

        boolean first = true;
        for (TIDProb entry : entries) {
            if (!first) {
                sb.append(", ");
            }
            sb.append(entry.toString());
            first = false;
        }

        sb.append("]");
        return sb.toString();
    }

    /**
     * TIDProb - Transaction ID and Probability pair.
     *
     * Immutable value class storing (tid, probability).
     * Implements Comparable for sorting by transaction ID.
     */
    public static class TIDProb implements Comparable<TIDProb> {

        /**
         * Transaction ID (index in database).
         */
        public final int tid;

        /**
         * Probability that item exists in this transaction.
         * Range: [0.0, 1.0]
         */
        public final double prob;

        /**
         * Constructor.
         *
         * @param tid  transaction ID
         * @param prob existence probability
         */
        public TIDProb(int tid, double prob) {
            this.tid = tid;
            this.prob = prob;
        }

        /**
         * Compare by transaction ID for sorting.
         * Required for merge-join intersection algorithm.
         *
         * @param other TIDProb to compare with
         * @return negative if this.tid < other.tid, etc.
         */
        @Override
        public int compareTo(TIDProb other) {
            return Integer.compare(this.tid, other.tid);
        }

        @Override
        public String toString() {
            return String.format("(%d, %.2f)", tid, prob);
        }
    }
}
