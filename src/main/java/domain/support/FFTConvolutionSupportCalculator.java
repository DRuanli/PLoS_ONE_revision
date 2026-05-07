package domain.support;

import domain.model.Tidset;
import shared.Constants;
import infrastructure.math.FFT;

/**
 * FFTConvolutionSupportCalculator - FFT-based probabilistic support calculator.
 *
 * Uses Fast Fourier Transform for polynomial multiplication.
 *
 * ═══════════════════════════════════════════════════════════════════════════════
 * ALGORITHM OVERVIEW
 * ═══════════════════════════════════════════════════════════════════════════════
 *
 * Same mathematical foundation as GFSupportCalculator:
 *   - Each transaction t contributes polynomial (1-p_t) + p_t·x
 *   - Product of all polynomials encodes probability distribution
 *   - Coefficient of x^s = P(support = s)
 *
 * Key Difference: HOW polynomials are multiplied
 *   - GF (DP): Multiply one at a time, O(n²) total
 *   - FFT: Divide-and-conquer with FFT, O(n log² n) total
 *
 * ═══════════════════════════════════════════════════════════════════════════════
 * WHY O(n log² n)?
 * ═══════════════════════════════════════════════════════════════════════════════
 *
 * Divide-and-Conquer Strategy:
 *
 *   Given n polynomials P₁, P₂, ..., Pₙ to multiply:
 *
 *   1. DIVIDE: Split into two groups
 *      Left  = P₁ × P₂ × ... × P_{n/2}
 *      Right = P_{n/2+1} × ... × Pₙ
 *
 *   2. CONQUER: Recursively multiply each group
 *
 *   3. COMBINE: Multiply Left × Right using FFT
 *
 *   Recurrence: T(n) = 2T(n/2) + O(n log n)
 *                    = O(n log² n)
 *
 *   Where O(n log n) is FFT multiplication of two degree-n polynomials.
 *
 * @author Dang Nguyen Le, Gia Huy Vo
 */
public class FFTConvolutionSupportCalculator extends AbstractSupportCalculator {

    /**
     * Constructor.
     *
     * @param tau probability threshold (0 < τ ≤ 1)
     */
    public FFTConvolutionSupportCalculator(double tau) {
        super(tau);
    }

    /**
     * Compute probabilistic support using FFT-based polynomial multiplication.
     *
     * @param transactionProbs probability array for all transactions
     * @param minRequiredSupport hint for early termination (unused)
     * @return probabilistic support value
     */
    @Override
    public int computeProbabilisticSupport(double[] transactionProbs, int minRequiredSupport) {
        // Step 1: Compute distribution via FFT
        double[] distribution = computeDistributionViaFFT(transactionProbs);

        // Step 2: Convert to frequentness (inherited from AbstractSupportCalculator)
        double[] frequentness = computeFrequentness(distribution);

        // Step 3: Binary search for max support (inherited)
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
        double[] distribution = computeDistributionViaFFT(transactionProbs);
        double[] frequentness = computeFrequentness(distribution);

        if (support >= 0 && support < frequentness.length) {
            return frequentness[support];
        }
        return 0.0;
    }

    /**
     * Get strategy name for logging.
     *
     * @return "Generating Functions (FFT)"
     */
    @Override
    public String getStrategyName() {
        return "Generating Functions (FFT)";
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
        double[] distribution = computeDistributionViaFFT(transactionProbs);
        double[] frequentness = computeFrequentness(distribution);

        int support = findProbabilisticSupport(frequentness);
        double probability = (support >= 0 && support < frequentness.length)
                           ? frequentness[support] : 0.0;

        return new double[]{support, probability};
    }

    /**
     * ═══════════════════════════════════════════════════════════════════════════
     * CORE: Compute distribution using FFT-based polynomial multiplication
     * ═══════════════════════════════════════════════════════════════════════════
     *
     * Algorithm:
     *   1. Filter out near-zero probabilities
     *   2. Convert each probability p to polynomial [1-p, p]
     *   3. Multiply all polynomials using divide-and-conquer + FFT
     *   4. Result coefficients = probability distribution
     *
     * @param transactionProbs probability array indexed by transaction ID
     * @return distribution where distribution[s] = P(support = s)
     */
    private double[] computeDistributionViaFFT(double[] transactionProbs) {
        // ─────────────────────────────────────────────────────────────
        // STEP 1: Filter out near-zero probabilities
        // Small probabilities contribute negligibly and may cause issues
        // ─────────────────────────────────────────────────────────────
        double[] filteredProbs = filterProbabilities(transactionProbs);

        // Handle empty case (no valid transactions)
        if (filteredProbs.length == 0) {
            // P(support = 0) = 1.0 (certainty of zero support)
            return new double[] {1.0};
        }

        // ─────────────────────────────────────────────────────────────
        // STEP 2: Convert each probability to polynomial coefficients
        //
        // Transaction with probability p contributes:
        //   (1-p) + p·x = [1-p, p] as coefficient array
        //
        // Meaning:
        //   - Coefficient of x⁰ (1-p): probability item NOT in transaction
        //   - Coefficient of x¹ (p):   probability item IN transaction
        // ─────────────────────────────────────────────────────────────
        double[][] polynomials = new double[filteredProbs.length][];

        for (int i = 0; i < filteredProbs.length; i++) {
            double p = filteredProbs[i];
            polynomials[i] = new double[] {1.0 - p, p};
        }

        // ─────────────────────────────────────────────────────────────
        // STEP 3: Multiply all polynomials using divide-and-conquer
        // ─────────────────────────────────────────────────────────────
        double[] result = multiplyAllPolynomials(polynomials);

        return result;
    }

    /**
     * Filter probabilities to remove near-zero and invalid values.
     *
     * @param probs input probability array
     * @return filtered array with only valid probabilities
     */
    private double[] filterProbabilities(double[] probs) {
        // Count valid probabilities
        int count = 0;
        for (double p : probs) {
            if (p >= Constants.MIN_PROB && p <= 1.0) {
                count++;
            }
        }

        // Create filtered array
        double[] filtered = new double[count];
        int idx = 0;

        for (double p : probs) {
            if (p >= Constants.MIN_PROB && p <= 1.0) {
                filtered[idx++] = p;
            }
        }

        return filtered;
    }

    /**
     * ═══════════════════════════════════════════════════════════════════════════
     * DIVIDE-AND-CONQUER: Multiply all polynomials
     * ═══════════════════════════════════════════════════════════════════════════
     *
     * Strategy:
     *   - If 0 polynomials: return [1.0] (identity)
     *   - If 1 polynomial: return it
     *   - Otherwise: divide, conquer, combine
     *
     * @param polynomials array of polynomial coefficient arrays
     * @return product of all polynomials
     */
    private double[] multiplyAllPolynomials(double[][] polynomials) {
        // Edge case: no polynomials
        if (polynomials.length == 0) {
            return new double[] {1.0};  // Multiplicative identity
        }

        // Base case: single polynomial
        if (polynomials.length == 1) {
            return polynomials[0];
        }

        // Recursive case: divide and conquer
        return multiplyRange(polynomials, 0, polynomials.length);
    }

    /**
     * ═══════════════════════════════════════════════════════════════════════════
     * RECURSIVE HELPER: Multiply polynomials in range [start, end)
     * ═══════════════════════════════════════════════════════════════════════════
     *
     * Divide-and-Conquer:
     *
     *   multiplyRange([P1, P2, P3, P4, P5, P6], 0, 6)
     *                          |
     *          ┌───────────────┴───────────────┐
     *          |                               |
     *   multiplyRange(0,3)              multiplyRange(3,6)
     *   = P1 × P2 × P3                  = P4 × P5 × P6
     *          |                               |
     *          └───────────┬───────────────────┘
     *                      |
     *               FFT.multiply(left, right)
     *
     * Time: T(n) = 2T(n/2) + O(n log n) = O(n log² n)
     *
     * @param polynomials all polynomials
     * @param start start index (inclusive)
     * @param end end index (exclusive)
     * @return product of polynomials[start..end-1]
     */
    private double[] multiplyRange(double[][] polynomials, int start, int end) {
        int count = end - start;

        // ─────────────────────────────────────────────────────────────
        // BASE CASE 1: Single polynomial
        // ─────────────────────────────────────────────────────────────
        if (count == 1) {
            return polynomials[start];
        }

        // ─────────────────────────────────────────────────────────────
        // BASE CASE 2: Two polynomials - multiply directly with FFT
        // ─────────────────────────────────────────────────────────────
        if (count == 2) {
            return FFT.multiplyPolynomials(polynomials[start], polynomials[start + 1]);
        }

        // ─────────────────────────────────────────────────────────────
        // RECURSIVE CASE: Divide in half
        // ─────────────────────────────────────────────────────────────
        int mid = start + count / 2;

        // CONQUER: Recursively multiply each half
        double[] left = multiplyRange(polynomials, start, mid);
        double[] right = multiplyRange(polynomials, mid, end);

        // COMBINE: Multiply the two results using FFT
        return FFT.multiplyPolynomials(left, right);
    }

    @Override
    public double[] computeProbabilisticSupportFromTidset(Tidset tidset, int totalTransactions) {
        if (tidset.isEmpty()) {
            return new double[]{0, 0.0};
        }

        // Extract only non-zero probabilities (sparse)
        int m = tidset.size();
        double[] probs = new double[m];
        int idx = 0;
        for (Tidset.TIDProb entry : tidset.getEntries()) {
            probs[idx++] = entry.prob;
        }

        // FFT on sparse array (size m, not totalTransactions)
        return computeProbabilisticSupportWithFrequentness(probs);
    }
}