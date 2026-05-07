package domain.support;

import domain.model.Tidset;
import shared.Constants;

/**
 * RecursiveConvolutionSupportCalculator - Hierarchical probabilistic support calculator.
 *
 * ═══════════════════════════════════════════════════════════════════════════════
 * ALGORITHM OVERVIEW
 * ═══════════════════════════════════════════════════════════════════════════════
 *
 * Unlike GF (sequential processing) or FFT (transform-based), this uses a
 * divide-and-conquer strategy:
 *
 * 1. DIVIDE: Split transactions into two halves
 * 2. CONQUER: Recursively compute distribution for each half
 * 3. COMBINE: Merge the two distributions via convolution
 *
 * The merge step uses:
 *   - Naive convolution: O(n²) per merge
 *
 * ═══════════════════════════════════════════════════════════════════════════════
 * COMPLEXITY ANALYSIS
 * ═══════════════════════════════════════════════════════════════════════════════
 *
 * Tree structure with log(n) levels:
 *   Level 0: 1 merge of size n
 *   Level 1: 2 merges of size n/2
 *   Level 2: 4 merges of size n/4
 *   ...
 *   Level log(n): n merges of size 1
 *
 * @author Dang Nguyen Le, Gia Huy Vo
 */
public class RecursiveConvolutionSupportCalculator extends AbstractSupportCalculator {

    /**
     * Constructor.
     *
     * @param tau probability threshold (0 < τ ≤ 1)
     */
    public RecursiveConvolutionSupportCalculator(double tau) {
        super(tau);
    }

    /**
     * Compute probabilistic support from dense probability array.
     *
     * @param transactionProbs probability array for all transactions
     * @param minRequiredSupport hint for early termination (unused in this implementation)
     * @return probabilistic support value
     */
    @Override
    public int computeProbabilisticSupport(double[] transactionProbs, int minRequiredSupport) {
        // ═════════════════════════════════════════════════════════════════
        // STEP 1: Compute probability distribution using Divide & Conquer
        // ═════════════════════════════════════════════════════════════════
        // Recursively split transactions and merge via convolution
        // Returns array where distribution[s] = P(support exactly equals s)
        // ═════════════════════════════════════════════════════════════════
        double[] distribution = divideAndConquer(transactionProbs, 0, transactionProbs.length);

        // ═════════════════════════════════════════════════════════════════
        // STEP 2: Convert distribution to frequentness (cumulative prob)
        // ═════════════════════════════════════════════════════════════════
        // frequentness[s] = P(support >= s) = sum of distribution[s..n]
        // Computed by cumulative sum from right to left
        // ═════════════════════════════════════════════════════════════════
        double[] frequentness = computeFrequentness(distribution);

        // ═════════════════════════════════════════════════════════════════
        // STEP 3: Find maximum support where P(support >= s) >= τ
        // ═════════════════════════════════════════════════════════════════
        // Uses binary search on frequentness array
        // Returns largest s such that frequentness[s] >= tau
        // ═════════════════════════════════════════════════════════════════
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
        double[] distribution = divideAndConquer(transactionProbs, 0, transactionProbs.length);
        double[] frequentness = computeFrequentness(distribution);

        // Return frequentness at requested support level
        if (support >= 0 && support < frequentness.length) {
            return frequentness[support];
        }
        return 0.0;
    }

    /**
     * Compute both support and probability in single call (optimized).
     *
     * @param transactionProbs probability array
     * @return [support, probability]
     */
    @Override
    public double[] computeProbabilisticSupportWithFrequentness(double[] transactionProbs) {
        // Single distribution computation
        double[] distribution = divideAndConquer(transactionProbs, 0, transactionProbs.length);
        double[] frequentness = computeFrequentness(distribution);

        // Extract both values
        int support = findProbabilisticSupport(frequentness);
        double probability = (support >= 0 && support < frequentness.length)
                           ? frequentness[support] : 0.0;

        return new double[]{support, probability};
    }

    /**
     * Compute support and probability from sparse tidset.
     *
     * @param tidset sparse transaction ID set with probabilities
     * @param totalTransactions total transactions (unused, for interface compatibility)
     * @return [support, probability]
     */
    @Override
    public double[] computeProbabilisticSupportFromTidset(Tidset tidset, int totalTransactions) {
        // ═════════════════════════════════════════════════════════════════
        // EMPTY TIDSET CHECK
        // ═════════════════════════════════════════════════════════════════
        // If itemset appears in zero transactions, support = 0, probability = 0
        // ═════════════════════════════════════════════════════════════════
        if (tidset.isEmpty()) {
            return new double[]{0, 0.0};
        }

        // ═════════════════════════════════════════════════════════════════
        // EXTRACT PROBABILITIES: Convert sparse Tidset to dense array
        // ═════════════════════════════════════════════════════════════════
        // Tidset stores only transactions where itemset appears
        // Extract probability values into dense array for D&C algorithm
        //
        // NOTE: This differs from GF's sparse optimization
        // GF processes Tidset entries directly (O(m²) algorithm)
        // D&C needs full array for recursive splitting (O(m² log m))
        // ═════════════════════════════════════════════════════════════════
        int m = tidset.size();  // Number of transactions containing itemset
        double[] probs = new double[m];  // Probability array
        int idx = 0;

        // Copy probabilities from tidset entries to array
        for (Tidset.TIDProb entry : tidset.getEntries()) {
            probs[idx++] = entry.prob;  // Probability of itemset in this transaction
        }

        // ═════════════════════════════════════════════════════════════════
        // COMPUTE: Run divide-and-conquer on extracted probabilities
        // ═════════════════════════════════════════════════════════════════
        // Uses same algorithm as dense version, but on m items instead of n
        // ═════════════════════════════════════════════════════════════════
        return computeProbabilisticSupportWithFrequentness(probs);
    }

    /**
     * Get strategy name for logging.
     *
     * @return strategy name
     */
    @Override
    public String getStrategyName() {
        return "Divide & Conquer (Hierarchical with Naive Convolution)";
    }

    /**
     * ═══════════════════════════════════════════════════════════════════════════
     * CORE ALGORITHM: Divide and Conquer Distribution Computation
     * ═══════════════════════════════════════════════════════════════════════════
     *
     * Recursively splits transactions and merges distributions.
     *
     * Algorithm:
     *   1. Base case: Single transaction → return [1-p, p]
     *   2. Divide: Split transactions in half
     *   3. Conquer: Recursively compute distribution for each half
     *   4. Combine: Convolve the two distributions
     *
     * Example with 4 transactions [0.6, 0.8, 0.5, 0.7]:
     *
     *                    [0.6, 0.8, 0.5, 0.7]
     *                     /                \
     *              [0.6, 0.8]            [0.5, 0.7]
     *               /      \              /      \
     *            [0.6]    [0.8]        [0.5]    [0.7]
     *             ↓         ↓            ↓        ↓
     *         [0.4,0.6] [0.2,0.8]   [0.5,0.5] [0.3,0.7]
     *              \      /              \      /
     *           convolve              convolve
     *               ↓                     ↓
     *        [0.08,0.44,0.48]      [0.15,0.50,0.35]
     *                 \                 /
     *                    convolve
     *                       ↓
     *            [0.012, 0.106, 0.320, 0.394, 0.168]
     *
     * @param probs probability array
     * @param start start index (inclusive)
     * @param end end index (exclusive)
     * @return probability distribution where result[s] = P(support = s)
     */
    private double[] divideAndConquer(double[] probs, int start, int end) {
        // Calculate number of transactions in this subproblem
        int length = end - start;

        // ─────────────────────────────────────────────────────────────────
        // BASE CASE: Single transaction
        // ─────────────────────────────────────────────────────────────────
        // A single transaction t with probability p contributes:
        //   P(support = 0) = 1 - p  (itemset NOT in transaction)
        //   P(support = 1) = p      (itemset IS in transaction)
        //
        // This is the leaf of the recursion tree where we return a simple
        // 2-element distribution representing a single Bernoulli trial.
        // ─────────────────────────────────────────────────────────────────
        if (length == 1) {
            // Extract probability of itemset appearing in this transaction
            double p = probs[start];

            // Skip near-zero probabilities for numerical stability
            // Very small p contributes negligibly and may cause underflow
            if (p < Constants.MIN_PROB) {
                // Treat as impossible: P(support=0) = 1, P(support=1) = 0
                return new double[]{1.0, 0.0};
            }

            // Return distribution: [P(not in transaction), P(in transaction)]
            return new double[]{1.0 - p, p};
        }

        // ─────────────────────────────────────────────────────────────────
        // DIVIDE: Split transaction range into two equal halves
        // ─────────────────────────────────────────────────────────────────
        // Classic divide-and-conquer split at midpoint
        // Left half:  [start, mid)   - first half of transactions
        // Right half: [mid, end)     - second half of transactions
        // ─────────────────────────────────────────────────────────────────
        int mid = start + length / 2;

        // ─────────────────────────────────────────────────────────────────
        // CONQUER: Recursively compute probability distributions
        // ─────────────────────────────────────────────────────────────────

        // Left subtree: compute distribution for transactions [start, mid)
        // This recursively processes the first half of transactions
        double[] leftDist = divideAndConquer(probs, start, mid);

        // Right subtree: compute distribution for transactions [mid, end)
        // This recursively processes the second half of transactions
        //
        // NOTE: This is where parallelization can happen!
        // In parallel implementation (see DCParallel.java):
        //   leftTask.fork()                  → Execute left asynchronously
        //   rightDist = rightTask.compute()  → Compute right in current thread
        //   leftDist = leftTask.join()       → Wait for left result
        //
        // Sequential version processes left then right in same thread
        double[] rightDist = divideAndConquer(probs, mid, end);

        // ─────────────────────────────────────────────────────────────────
        // COMBINE: Merge distributions via discrete convolution
        // ─────────────────────────────────────────────────────────────────
        // Since left and right transaction sets are independent,
        // the distribution of their sum is the convolution of their
        // individual distributions.
        //
        // Mathematical justification:
        //   If X = support from left half, Y = support from right half
        //   Then total support S = X + Y (independent random variables)
        //   P(S = s) = Σ_i P(X = i) × P(Y = s-i)  (convolution formula)
        // ─────────────────────────────────────────────────────────────────
        return convolve(leftDist, rightDist);
    }

    /**
     * ═══════════════════════════════════════════════════════════════════════════
     * CONVOLUTION: Merge two probability distributions
     * ═══════════════════════════════════════════════════════════════════════════
     *
     * Given two independent random variables with distributions:
     *   a[i] = P(left = i)
     *   b[j] = P(right = j)
     *
     * Their sum has distribution:
     *   result[s] = P(left + right = s) = Σ_i a[i] × b[s-i]
     *
     * This is the DEFINITION of discrete convolution.
     *
     * Example: a = [0.4, 0.6], b = [0.2, 0.8]
     *   result[0] = 0.4 × 0.2 = 0.08
     *   result[1] = 0.4 × 0.8 + 0.6 × 0.2 = 0.44
     *   result[2] = 0.6 × 0.8 = 0.48
     *   Result: [0.08, 0.44, 0.48]
     *
     * @param a first distribution
     * @param b second distribution
     * @return convolution of a and b
     */
    private double[] convolve(double[] a, double[] b) {
        return naiveConvolve(a, b);
    }

    /**
     * ═══════════════════════════════════════════════════════════════════════════
     * NAIVE CONVOLUTION: Direct computation of discrete convolution
     * ═══════════════════════════════════════════════════════════════════════════
     *
     * Computes the convolution of two probability distributions using the
     * definition of discrete convolution:
     *
     *   result[s] = Σ_{i=0}^{s} a[i] × b[s-i]
     *
     * Mathematical Interpretation:
     *   If a[i] = P(X = i) and b[j] = P(Y = j) where X and Y are independent,
     *   then result[s] = P(X + Y = s)
     *
     * Example: a = [0.4, 0.6], b = [0.2, 0.8]
     *   result[0] = a[0] × b[0] = 0.4 × 0.2 = 0.08
     *   result[1] = a[0] × b[1] + a[1] × b[0] = 0.4 × 0.8 + 0.6 × 0.2 = 0.44
     *   result[2] = a[1] × b[1] = 0.6 × 0.8 = 0.48
     *   Result: [0.08, 0.44, 0.48]
     *
     * Time Complexity: O(|a| × |b|)
     *   - Nested loops iterate over all pairs (i, j)
     *   - Total iterations: |a| × |b|
     *
     * Space Complexity: O(|a| + |b| - 1)
     *   - Result array size: max possible sum is (|a|-1) + (|b|-1)
     *
     * Advantages:
     *   - Simple, no overhead
     *   - Easy to understand and maintain
     *   - Exact (no numerical errors from FFT)
     *
     * Disadvantages:
     *   - Slower than FFT-based convolution for large arrays
     *   - O(n²) vs O(n log n) for FFT
     *
     * @param a first probability distribution where a[i] = P(X = i)
     * @param b second probability distribution where b[j] = P(Y = j)
     * @return convolution result where result[s] = P(X + Y = s)
     */
    private double[] naiveConvolve(double[] a, double[] b) {
        // Get lengths of input distributions
        int lenA = a.length;
        int lenB = b.length;

        // ═════════════════════════════════════════════════════════════════
        // RESULT ARRAY SIZE
        // ═════════════════════════════════════════════════════════════════
        // Maximum sum occurs when X = |a|-1 and Y = |b|-1
        // Therefore, result needs (|a|-1) + (|b|-1) + 1 = |a| + |b| - 1 elements
        //
        // Example: a.length=3 (indices 0,1,2), b.length=2 (indices 0,1)
        //   Max sum = 2 + 1 = 3, so result needs indices 0,1,2,3 → size 4
        //   Formula: 3 + 2 - 1 = 4 ✓
        // ═════════════════════════════════════════════════════════════════
        double[] result = new double[lenA + lenB - 1];

        // ═════════════════════════════════════════════════════════════════
        // DOUBLE LOOP: Compute convolution via definition
        // ═════════════════════════════════════════════════════════════════
        // For each pair (i, j):
        //   - a[i] represents P(left support = i)
        //   - b[j] represents P(right support = j)
        //   - Their product contributes to P(total support = i+j)
        //
        // We accumulate all contributions to each result[s]
        // ═════════════════════════════════════════════════════════════════

        // Iterate over all elements in first distribution
        for (int i = 0; i < lenA; i++) {
            // Iterate over all elements in second distribution
            for (int j = 0; j < lenB; j++) {
                // ─────────────────────────────────────────────────────────
                // ACCUMULATE: Add contribution to result[i+j]
                // ─────────────────────────────────────────────────────────
                // P(total = i+j) includes the case where:
                //   left contributes i AND right contributes j
                // Probability of this case: a[i] × b[j] (independence)
                // ─────────────────────────────────────────────────────────
                result[i + j] += a[i] * b[j];
            }
        }

        // Return complete convolution
        return result;
    }
}
