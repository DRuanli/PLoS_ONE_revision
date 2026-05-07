package domain.support;

import domain.model.Tidset;

/**
 * SupportCalculator - Strategy Pattern interface for probabilistic support computation.
 *
 * <p><b>Uncertain Database Model:</b></p>
 * <p>In uncertain databases, items have existential probabilities. Support is not a
 * deterministic count but a probabilistic measure computed using generating functions.</p>
 *
 * <p><b>Mathematical Foundation:</b></p>
 * <ul>
 *   <li><b>Existence Probability:</b> P(i,t) ∈ [0,1] - probability item i exists in transaction t</li>
 *   <li><b>Itemset Probability:</b> P(X ⊆ t) = ∏ P(i,t) for all i ∈ X</li>
 *   <li><b>Support Distribution:</b> P(sup(X) = s) computed via generating functions</li>
 *   <li><b>Frequentness:</b> P(sup(X) ≥ s) - tail probability (cumulative)</li>
 *   <li><b>Probabilistic Support:</b> ProbSup_τ(X) = max{s : P(sup(X) ≥ s) ≥ τ}</li>
 * </ul>
 *
 * <p><b>Generating Function Method:</b></p>
 * <p>G(x) = ∏[(1 - p_t) + p_t·x] where coefficient of x^s gives P(sup(X) = s)</p>
 *
 * <p><b>Academic References:</b></p>
 * <ul>
 *   <li>Chui et al. - Mining frequent itemsets from uncertain data</li>
 *   <li>Bernecker et al. - Probabilistic frequent itemset mining in uncertain databases</li>
 *   <li>Aggarwal et al. - Frequent pattern mining with uncertain data</li>
 * </ul>
 *
 * @author Dang Nguyen Le, Gia Huy Vo
 */
public interface SupportCalculator {

    /**
     * Computes only the probabilistic support value (without frequentness).
     *
     * <p><b>Definition:</b> ProbSup_τ(X) = max{s : P(sup(X) ≥ s) ≥ τ}</p>
     *
     * <p><b>Note:</b> {@link #computeProbabilisticSupportWithFrequentness(double[])}
     * or {@link #computeProbabilisticSupportFromTidset(Tidset, int)} are preferred
     * as they return both support and frequentness in a single computation.</p>
     *
     * @param existenceProbabilities array where existenceProbabilities[t] = P(itemset ⊆ transaction t)
     * @param minRequiredSupport hint for early termination (implementation-specific, may be ignored)
     * @return the probabilistic support value (maximum s where frequentness ≥ τ)
     */
    int computeProbabilisticSupport(double[] existenceProbabilities, int minRequiredSupport);

    /**
     * Computes frequentness (tail probability) at a given support level.
     *
     * <p><b>Definition:</b> Frequentness at level s = P(sup(X) ≥ s)</p>
     *
     * <p>This is the cumulative probability that the itemset appears in at least s transactions.</p>
     *
     * <p><b>Note:</b> {@link #computeProbabilisticSupportWithFrequentness(double[])}
     * is preferred when both support and frequentness are needed.</p>
     *
     * @param existenceProbabilities array of transaction existence probabilities
     * @param supportLevel target support level s
     * @return P(sup(X) ≥ supportLevel) - probability of achieving at least that support
     */
    double computeProbability(double[] existenceProbabilities, int supportLevel);

    /**
     * Get name of this calculator strategy (for logging/debugging).
     *
     * @return human-readable strategy name
     */
    String getStrategyName();

    /**
     * Computes probabilistic support and frequentness in a single operation.
     *
     * <p><b>Definition:</b></p>
     * <ul>
     *   <li><b>Probabilistic Support:</b> ProbSup_τ(X) = max{s : P(sup(X) ≥ s) ≥ τ}</li>
     *   <li><b>Frequentness:</b> P(sup(X) ≥ ProbSup_τ(X)) - the actual probability at that support level</li>
     * </ul>
     *
     * <p><b>Why Combined Computation:</b></p>
     * <p>Computing both values together is more efficient than separate calls because
     * the support distribution P(sup(X) = s) is computed only once. The frequentness
     * is obtained as a byproduct of finding the probabilistic support threshold.</p>
     *
     * <p><b>Algorithm:</b></p>
     * <ol>
     *   <li>Compute support distribution using generating functions: G(x) = ∏[(1-p_t) + p_t·x]</li>
     *   <li>Extract coefficients to get P(sup(X) = s) for each s</li>
     *   <li>Compute cumulative probabilities (frequentness): P(sup(X) ≥ s)</li>
     *   <li>Find maximum s where frequentness ≥ τ (the probabilistic support)</li>
     *   <li>Return both the support value and its corresponding frequentness</li>
     * </ol>
     *
     * @param existenceProbabilities array where existenceProbabilities[t] = P(itemset ⊆ transaction t)
     * @return array [probabilisticSupport, frequentness] where:
     *         <ul>
     *           <li>probabilisticSupport = max{s : P(sup(X) ≥ s) ≥ τ} (integer, returned as double)</li>
     *           <li>frequentness = P(sup(X) ≥ probabilisticSupport) ∈ [τ, 1]</li>
     *         </ul>
     */
    double[] computeProbabilisticSupportWithFrequentness(double[] existenceProbabilities);

    /**
     * Computes probabilistic support and frequentness from sparse Tidset representation.
     *
     * <p><b>Input Format:</b></p>
     * <p>Tidset (Transaction ID Set) is a sparse representation containing only
     * transactions where the itemset has non-zero probability. This avoids storing
     * zeros for transactions where the itemset cannot appear.</p>
     *
     * <p><b>Memory Efficiency:</b></p>
     * <p>For sparse itemsets (appearing in few transactions), this method is much
     * more memory-efficient than dense array representation:</p>
     * <ul>
     *   <li><b>Dense:</b> O(totalTransactions) array - e.g., 100,000 doubles = 800 KB</li>
     *   <li><b>Sparse:</b> O(tidset.size()) - e.g., 500 entries = ~4 KB</li>
     *   <li><b>Savings:</b> 200x less memory for 0.5% sparsity</li>
     * </ul>
     *
     * <p><b>Implementation Note:</b></p>
     * <p>Default implementation converts Tidset to dense array and delegates to
     * {@link #computeProbabilisticSupportWithFrequentness(double[])}. Subclasses
     * can override for direct sparse computation without materialization.</p>
     *
     * <p><b>Equivalence:</b></p>
     * <p>This method computes mathematically identical results to the dense version.
     * Only the input representation differs - the generating function algorithm
     * processes the same probabilities.</p>
     *
     * @param tidset sparse transaction ID set with existence probabilities
     * @param databaseSize total number of transactions in database
     * @return array [probabilisticSupport, frequentness] - same format as
     *         {@link #computeProbabilisticSupportWithFrequentness(double[])}
     */
    default double[] computeProbabilisticSupportFromTidset(Tidset tidset, int databaseSize) {
        // Default: convert sparse tidset to dense probability array
        // Subclasses can override for memory-efficient sparse computation
        double[] existenceProbabilities = tidset.toTransactionProbabilities(databaseSize);

        // Delegate to dense computation
        return computeProbabilisticSupportWithFrequentness(existenceProbabilities);
    }
}