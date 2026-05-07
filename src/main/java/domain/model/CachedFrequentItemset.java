package domain.model;

import infrastructure.persistence.Vocabulary;
import java.util.BitSet;

/**
 * CachedFrequentItemset - A FrequentItemset with cached Tidset.
 *
 * Hierarchy:
 *   Itemset (just items)
 *       └── FrequentItemset (+ support, probability)
 *               └── CachedFrequentItemset (+ tidset)  ← THIS CLASS
 *
 * A CachedFrequentItemset IS-A FrequentItemset that also stores:
 *   - tidset: cached transaction IDs and probabilities
 *
 * The tidset is used for:
 *   - Efficient intersection when computing superset support
 *   - Avoiding repeated database scans
 *
 * Used internally during mining (not in final results).
 *
 * @author Dang Nguyen Le, Gia Huy Vo
 */
public class CachedFrequentItemset extends FrequentItemset {

    /**
     * Cached tidset for efficient intersection.
     *
     * Contains all transactions where this itemset appears,
     * along with the probability of the itemset in each transaction.
     * Enables O(n) intersection instead of O(n×m) database scans.
     */
    private final Tidset tidset;

    /**
     * Constructor from base Itemset with all data.
     *
     * @param base existing Itemset to copy
     * @param support probabilistic support
     * @param probability P(support >= support)
     * @param tidset cached transaction ID set
     */
    public CachedFrequentItemset(Itemset base, int support,
                                  double probability, Tidset tidset) {
        super(base, support, probability);
        this.tidset = tidset;
    }

    /**
     * Constructor from FrequentItemset adding tidset.
     *
     * Useful when you already have a FrequentItemset and want to cache it.
     *
     * @param frequent existing FrequentItemset
     * @param tidset tidset to cache
     */
    public CachedFrequentItemset(FrequentItemset frequent, Tidset tidset) {
        super(frequent, frequent.getSupport(), frequent.getProbability());
        this.tidset = tidset;
    }

    /**
     * Protected constructor from BitSet (for internal use).
     *
     * @param items BitSet containing item indices
     * @param vocab vocabulary for item name lookup
     * @param support probabilistic support value
     * @param probability P(support >= support)
     * @param tidset cached transaction ID set
     */
    protected CachedFrequentItemset(BitSet items, Vocabulary vocab,
                                    int support, double probability,
                                    Tidset tidset) {
        super(items, vocab, support, probability);
        this.tidset = tidset;
    }

    /**
     * Get cached tidset.
     *
     * @return tidset containing transaction IDs and probabilities
     */
    public Tidset getTidset() {
        return tidset;
    }

    /**
     * Convert to FrequentItemset (strip tidset).
     *
     * Use this when returning final results - tidset not needed
     * and takes memory.
     *
     * @return FrequentItemset without tidset cache
     */
    public FrequentItemset toFrequentItemset() {
        return new FrequentItemset(this, getSupport(), getProbability());
    }

    /**
     * Create union with another itemset, with new metrics and tidset.
     *
     * @param other itemset to union with
     * @param newSupport support of the union
     * @param newProbability probability of the union
     * @param newTidset tidset of the union
     * @return new CachedFrequentItemset with combined items, new metrics, and new tidset
     */
    public CachedFrequentItemset unionWithCache(Itemset other,
                                                 int newSupport,
                                                 double newProbability,
                                                 Tidset newTidset) {
        BitSet result = (BitSet) getItemsBitSet().clone();
        result.or(other.getItemsBitSet());
        return new CachedFrequentItemset(result, getVocabulary(),
                                         newSupport, newProbability, newTidset);
    }

    /**
     * String representation with items, metrics, and tidset size.
     *
     * Format: {items} [sup=X, prob=Y.YYY, tidset_size=Z]
     *
     * @return human-readable representation
     */
    @Override
    public String toString() {
        return String.format("%s [sup=%d, prob=%.3f, tidset_size=%d]",
            toStringWithCodec(), getSupport(), getProbability(),
            tidset != null ? tidset.size() : 0);
    }
}
