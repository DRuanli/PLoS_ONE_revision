package domain.support;

import domain.model.Tidset;
import shared.Constants;

/**
 * DirectConvolutionSupportCalculator - Generating Function based probabilistic support calculator.
 *
 * Uses Dynamic Programming to compute probability distribution via polynomial multiplication.
 *
 * ═══════════════════════════════════════════════════════════════════════════════
 * THEORETICAL FOUNDATION: GENERATING FUNCTIONS
 * ═══════════════════════════════════════════════════════════════════════════════
 *
 * In uncertain databases, each transaction t has probability p_t that itemset X appears.
 * Total support S is sum of independent Bernoulli random variables.
 *
 * Key Insight: Each transaction contributes polynomial (1-p) + p·x where:
 *   - (1-p): probability item NOT in transaction (support unchanged)
 *   - p·x:   probability item IN transaction (support increases by 1)
 *
 * Product of all polynomials encodes the probability distribution:
 *   G(x) = Π[(1-p_t) + p_t·x] for all transactions t
 *
 * Coefficient of x^s in G(x) = P(support = s)
 *
 * Example with 3 transactions, probabilities [0.6, 0.8, 0.5]:
 *   G(x) = (0.4 + 0.6x)(0.2 + 0.8x)(0.5 + 0.5x)
 *        = 0.04 + 0.28x + 0.44x² + 0.24x³
 *
 *   P(support = 0) = 0.04
 *   P(support = 1) = 0.28
 *   P(support = 2) = 0.44
 *   P(support = 3) = 0.24
 *
 * ═══════════════════════════════════════════════════════════════════════════════
 * ALGORITHM: DYNAMIC PROGRAMMING POLYNOMIAL MULTIPLICATION
 * ═══════════════════════════════════════════════════════════════════════════════
 *
 * Instead of multiplying all polynomials then extracting coefficients,
 * we incrementally update coefficient array:
 *
 * Initial: coeffs = [1, 0, 0, ...] (P(support=0) = 1 before any transaction)
 *
 * For each transaction with probability p:
 *   new_coeffs[i] = old_coeffs[i] × (1-p) + old_coeffs[i-1] × p
 *
 * This represents: P(support = i after t transactions) =
 *   P(support = i after t-1) × P(not in t) +
 *   P(support = i-1 after t-1) × P(in t)
 *
 *
 * @author Dang Nguyen Le, Gia Huy Vo
 */
public class DirectConvolutionSupportCalculator extends AbstractSupportCalculator {

    /**
     * Constructor.
     *
     * @param tau probability threshold (0 < τ ≤ 1)
     */
    public DirectConvolutionSupportCalculator(double tau) {
        // Call parent constructor for validation
        super(tau);
    }

    /**
     * Compute probabilistic support from dense probability array.
     *
     * @param transactionProbs probability array for all transactions
     * @param minRequiredSupport hint for early termination (currently unused)
     * @return probabilistic support value
     */
    @Override
    public int computeProbabilisticSupport(double[] transactionProbs, int minRequiredSupport) {
        // Step 1: Compute distribution via generating function
        double[] distribution = computeDistributionViaGeneratingFunction(transactionProbs);

        // Step 2: Convert to frequentness (cumulative probabilities)
        double[] frequentness = computeFrequentness(distribution);

        // Step 3: Binary search for max support where freq[s] ≥ τ
        return findProbabilisticSupport(frequentness);
    }

    /**
     * Compute probability P(support ≥ s) for given support value.
     *
     * @param transactionProbs probability array
     * @param support target support value
     * @return P(support ≥ s)
     */
    @Override
    public double computeProbability(double[] transactionProbs, int support) {
        // Compute full distribution
        double[] distribution = computeDistributionViaGeneratingFunction(transactionProbs);
        double[] frequentness = computeFrequentness(distribution);

        // Return frequentness at requested support level
        if (support >= 0 && support < frequentness.length) {
            return frequentness[support];
        }
        return 0.0;
    }

    /**
     * Get strategy name for logging.
     *
     * @return "Generating Functions (Polynomial DP)"
     */
    @Override
    public String getStrategyName() {
        return "Generating Functions (Polynomial DP)";
    }

    /**
     * Compute both support and probability in single call (optimized).
     *
     * More efficient than separate calls because distribution computed once.
     *
     * @param transactionProbs probability array
     * @return [support, probability]
     */
    @Override
    public double[] computeProbabilisticSupportWithFrequentness(double[] transactionProbs) {
        // Single distribution computation
        double[] distribution = computeDistributionViaGeneratingFunction(transactionProbs);
        double[] frequentness = computeFrequentness(distribution);

        // Extract both values
        int support = findProbabilisticSupport(frequentness);
        double probability = (support >= 0 && support < frequentness.length)
                           ? frequentness[support] : 0.0;

        return new double[]{support, probability};
    }

    /**
     * ═══════════════════════════════════════════════════════════════════════════
     * CORE ALGORITHM: Compute probability distribution via Generating Function
     * ═══════════════════════════════════════════════════════════════════════════
     *
     * Uses in-place Dynamic Programming to multiply polynomials.
     *
     * Key Recurrence:
     *   coeffs[i] = coeffs[i] × (1-p) + coeffs[i-1] × p
     *
     * CRITICAL: Must iterate BACKWARD (i = degree+1 down to 1)
     *
     * Why backward iteration?
     *   - coeffs[i] depends on coeffs[i-1] (the OLD value)
     *   - Forward iteration would overwrite coeffs[i-1] before using it
     *   - Backward iteration uses coeffs[i-1] before it's updated
     *
     * Example: coeffs = [0.5, 0.5, 0], p = 0.8
     *
     *   WRONG (forward):
     *     i=0: coeffs[0] = 0.5 × 0.2 = 0.1
     *     i=1: coeffs[1] = 0.5 × 0.2 + 0.1 × 0.8 = 0.18  ← WRONG! Used updated coeffs[0]
     *
     *   CORRECT (backward):
     *     i=2: coeffs[2] = 0 × 0.2 + 0.5 × 0.8 = 0.4
     *     i=1: coeffs[1] = 0.5 × 0.2 + 0.5 × 0.8 = 0.5  ← Correct! Used original coeffs[0]
     *     i=0: coeffs[0] = 0.5 × 0.2 = 0.1
     *
     * Time Complexity: O(n²) - for n transactions, update up to n coefficients each
     * Space Complexity: O(n) - single coefficient array
     *
     * @param transactionProbs probability array indexed by transaction ID
     * @return distribution array where distribution[s] = P(support = s)
     */
    private double[] computeDistributionViaGeneratingFunction(double[] transactionProbs) {
        int n = transactionProbs.length;

        // Coefficient array: coeffs[s] = P(support = s)
        // Size n+1 because support can range from 0 to n
        double[] coeffs = new double[n + 1];

        // Initial state: before any transaction, P(support = 0) = 1
        coeffs[0] = 1.0;

        // Track current polynomial degree (number of processed transactions)
        int degree = 0;

        // Process each transaction
        for (int t = 0; t < n; t++) {
            double p = transactionProbs[t];

            // Skip near-zero probabilities for numerical stability
            // Very small p contributes negligibly and may cause underflow
            if (p < Constants.MIN_PROB) continue;

            // ═══════════════════════════════════════════════════════════════
            // POLYNOMIAL MULTIPLICATION: coeffs × [(1-p) + p·x]
            // ═══════════════════════════════════════════════════════════════
            // New coefficient at position i:
            //   new_coeffs[i] = old_coeffs[i] × (1-p)     ← item NOT in transaction
            //                 + old_coeffs[i-1] × p       ← item IN transaction
            //
            // BACKWARD ITERATION: i from (degree+1) down to 1
            // Ensures we use original coeffs[i-1] before updating it
            // ═══════════════════════════════════════════════════════════════

            for (int i = degree + 1; i >= 1; i--) {
                // coeffs[i] = P(support = i) × P(not in t) + P(support = i-1) × P(in t)
                coeffs[i] = coeffs[i] * (1.0 - p) + coeffs[i - 1] * p;
            }

            // Special case: coeffs[0] has no coeffs[-1] term
            // P(support = 0) = P(was 0) × P(not in transaction)
            coeffs[0] = coeffs[0] * (1.0 - p);

            // Polynomial degree increases by 1
            degree++;
        }

        return coeffs;
    }

    /**
     * ═══════════════════════════════════════════════════════════════════════════
     * SPARSE OPTIMIZATION: Compute directly from Tidset without dense array
     * ═══════════════════════════════════════════════════════════════════════════
     *
     * Memory optimization for sparse itemsets (common in real data).
     *
     * Dense approach: Allocate array[totalTransactions], mostly zeros
     * Sparse approach: Only process tidset.size() entries
     *
     * Example: 100,000 transactions, itemset in 500
     *   Dense:  double[100001] = 800KB, process 100,000 entries
     *   Sparse: double[501] = 4KB, process 500 entries
     *
     * Time Complexity: O(m²) where m = tidset size (vs O(n²) for dense)
     * Space Complexity: O(m) (vs O(n) for dense)
     *
     * @param tidset sparse transaction ID set with probabilities
     * @param totalTransactions total transactions (unused in sparse, for interface)
     * @return [support, probability]
     */
    @Override
    public double[] computeProbabilisticSupportFromTidset(Tidset tidset, int totalTransactions) {
        // Handle empty tidset
        if (tidset.isEmpty()) {
            return new double[]{0, 0.0};
        }

        int m = tidset.size();

        // ═══════════════════════════════════════════════════════════════
        // SPARSE DP: Same algorithm, but only m+1 coefficients needed
        // ═══════════════════════════════════════════════════════════════
        double[] coeffs = new double[m + 1];
        coeffs[0] = 1.0;  // Initial: P(support = 0) = 1
        int degree = 0;

        // Process only transactions in tidset (not all transactions)
        for (Tidset.TIDProb entry : tidset.getEntries()) {
            double p = entry.prob;

            // Skip near-zero probabilities
            if (p < Constants.MIN_PROB) continue;

            // Same backward DP as dense version
            for (int i = degree + 1; i >= 1; i--) {
                coeffs[i] = coeffs[i] * (1.0 - p) + coeffs[i - 1] * p;
            }
            coeffs[0] = coeffs[0] * (1.0 - p);
            degree++;
        }

        // ═══════════════════════════════════════════════════════════════
        // INLINE FREQUENTNESS: Avoid separate method call overhead
        // ═══════════════════════════════════════════════════════════════
        double[] frequentness = new double[degree + 1];

        // Cumulative sum from right to left
        frequentness[degree] = coeffs[degree];
        for (int i = degree - 1; i >= 0; i--) {
            frequentness[i] = frequentness[i + 1] + coeffs[i];
        }

        // ═══════════════════════════════════════════════════════════════
        // INLINE BINARY SEARCH: Find max support where freq[s] ≥ τ
        // ═══════════════════════════════════════════════════════════════
        int support = 0;
        int left = 0, right = degree;

        while (left <= right) {
            int mid = left + (right - left) / 2;

            if (frequentness[mid] >= tau - Constants.EPSILON) {
                support = mid;      // Valid support, try larger
                left = mid + 1;
            } else {
                right = mid - 1;    // Invalid, try smaller
            }
        }

        // Get probability at computed support level
        double probability = (support >= 0 && support <= degree)
                           ? frequentness[support] : 0.0;

        return new double[]{support, probability};
    }
}
