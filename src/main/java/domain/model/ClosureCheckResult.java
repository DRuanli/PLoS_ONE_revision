package domain.model;

import java.util.List;
import java.util.Collections;

/**
 * ClosureCheckResult - Result of closure checking operation.
 *
 * Contains:
 * - Whether the itemset is closed
 * - List of valid extensions to explore
 *
 * This is a simple immutable result object used by the TUFCI algorithm.
 *
 * A closed itemset is one where no superset has the same support.
 * During closure checking, we also generate valid extensions (supersets)
 * that should be explored next in the mining process.
 *
 * @author Dang Nguyen Le, Gia Huy Vo
 */
public class ClosureCheckResult {

    /**
     * True if the itemset is closed (no superset with same support).
     */
    private final boolean isClosed;

    /**
     * List of valid extension candidates to explore next.
     * These are supersets of the checked itemset that meet the minimum support threshold.
     */
    private final List<FrequentItemset> extensions;

    /**
     * Constructor.
     *
     * @param isClosed true if the itemset is closed
     * @param extensions list of valid extensions (will be made unmodifiable)
     */
    public ClosureCheckResult(boolean isClosed, List<FrequentItemset> extensions) {
        this.isClosed = isClosed;
        this.extensions = Collections.unmodifiableList(extensions);
    }

    /**
     * Check if the itemset is closed.
     *
     * @return true if closed (no superset has same support), false otherwise
     */
    public boolean isClosed() {
        return isClosed;
    }

    /**
     * Get list of valid extensions.
     *
     * @return unmodifiable list of FrequentItemset extensions
     */
    public List<FrequentItemset> getExtensions() {
        return extensions;
    }
}
