package domain.model;

import infrastructure.persistence.Vocabulary;
import java.util.BitSet;

/**
 * FrequentItemset - An Itemset with support and probability metrics.
 *
 * Hierarchy:
 *   Itemset (just items)
 *       └── FrequentItemset (+ support, probability)  ← THIS CLASS
 *               └── CachedFrequentItemset (+ tidset)
 *
 * A FrequentItemset IS-A Itemset that has been mined and has:
 *   - support: probabilistic support value
 *   - probability: P(support >= computed_support)
 *
 * Used for:
 *   - Final mining results (output)
 *   - Candidate patterns during mining
 *   - Priority queue entries
 *   - TopK heap entries
 *
 * @author Dang Nguyen Le, Gia Huy Vo
 */
public class FrequentItemset extends Itemset {

    /**
     * Probabilistic support of this itemset.
     *
     * Definition: ProbSupport_τ(X) = max{s : P(support(X) ≥ s) ≥ τ}
     *
     * In uncertain databases, each item has existence probability.
     * Support is the maximum count s such that the probability
     * of having at least s occurrences is >= threshold τ.
     */
    private final int support;

    /**
     * Probability that itemset has at least 'support' occurrences.
     *
     * Definition: P(support(X) ≥ support)
     *
     * This is the frequentness value at the computed support level.
     * Always >= τ (threshold) by definition of probabilistic support.
     */
    private final double probability;

    /**
     * Constructor from base Itemset with metrics.
     *
     * This is the PRIMARY constructor - creates FrequentItemset by
     * copying an existing Itemset and adding metrics.
     *
     * @param base existing Itemset to copy
     * @param support probabilistic support value
     * @param probability P(support >= support)
     */
    public FrequentItemset(Itemset base, int support, double probability) {
        super(base);  // Copy itemset data
        this.support = support;
        this.probability = probability;
    }

    /**
     * Constructor with vocabulary for creating new itemset with metrics.
     *
     * Creates empty itemset - items must be added via add().
     * Prefer using the Itemset-based constructor when possible.
     *
     * @param vocab vocabulary for item name lookup
     * @param support probabilistic support value
     * @param probability P(support >= support)
     */
    public FrequentItemset(Vocabulary vocab, int support, double probability) {
        super(vocab);
        this.support = support;
        this.probability = probability;
    }

    /**
     * Protected constructor from BitSet (for subclasses and union operations).
     *
     * @param items BitSet containing item indices
     * @param vocab vocabulary for item name lookup
     * @param support probabilistic support value
     * @param probability P(support >= support)
     */
    protected FrequentItemset(BitSet items, Vocabulary vocab,
                              int support, double probability) {
        super(items, vocab);
        this.support = support;
        this.probability = probability;
    }

    /**
     * Get probabilistic support.
     *
     * @return support value
     */
    public int getSupport() {
        return support;
    }

    /**
     * Get probability.
     *
     * @return probability of achieving this support level
     */
    public double getProbability() {
        return probability;
    }

    /**
     * Create union with another itemset.
     *
     * Note: Returns FrequentItemset with THIS itemset's metrics.
     * For proper union with new metrics, use unionWithMetrics() instead.
     *
     * @param other itemset to union with
     * @return new FrequentItemset with combined items and this's metrics
     */
    @Override
    public FrequentItemset union(Itemset other) {
        BitSet result = (BitSet) getItemsBitSet().clone();
        result.or(other.getItemsBitSet());
        return new FrequentItemset(result, getVocabulary(), support, probability);
    }

    /**
     * Create union with specific metrics.
     *
     * This is the preferred method when you know the new metrics.
     *
     * @param other itemset to union with
     * @param newSupport support of the union
     * @param newProbability probability of the union
     * @return new FrequentItemset with combined items and new metrics
     */
    public FrequentItemset unionWithMetrics(Itemset other,
                                            int newSupport,
                                            double newProbability) {
        BitSet result = (BitSet) getItemsBitSet().clone();
        result.or(other.getItemsBitSet());
        return new FrequentItemset(result, getVocabulary(), newSupport, newProbability);
    }

    /**
     * String representation with items and metrics.
     *
     * Format: {items} [sup=X, prob=Y.YYY]
     *
     * @return human-readable representation
     */
    @Override
    public String toString() {
        return String.format("%s [sup=%d, prob=%.3f]",
            toStringWithCodec(), support, probability);
    }

    /**
     * Compare by support (for sorting/priority queue).
     * Higher support = more important.
     *
     * @param a first itemset
     * @param b second itemset
     * @return comparison result (descending by support, then probability)
     */
    public static int compareBySupport(FrequentItemset a, FrequentItemset b) {
        // Primary: support descending (higher is better)
        int cmp = Integer.compare(b.support, a.support);
        if (cmp != 0) return cmp;

        // Secondary: probability descending (higher is better)
        return Double.compare(b.probability, a.probability);
    }

    /**
     * Compare for priority queue (support DESC, size ASC, probability DESC).
     *
     * This ordering ensures we explore the most promising candidates first:
     * - Higher support itemsets first (more frequent)
     * - Smaller itemsets first (when support is equal)
     * - Higher probability first (when support and size are equal)
     *
     * @param a first itemset
     * @param b second itemset
     * @return comparison result for priority queue ordering
     */
    public static int compareForPriorityQueue(FrequentItemset a, FrequentItemset b) {
        // Primary: support descending (higher support first)
        int cmp = Integer.compare(b.support, a.support);
        if (cmp != 0) return cmp;

        // Secondary: size ascending (smaller itemsets first)
        cmp = Integer.compare(a.size(), b.size());
        if (cmp != 0) return cmp;

        // Tertiary: probability descending (higher probability first)
        return Double.compare(b.probability, a.probability);
    }
}
