package domain.support;

import shared.Constants;

/**
 * AbstractSupportCalculator - Base class for support computation strategies.
 *
 * Template Method Pattern: Provides common functionality shared by all calculators:
 *   - computeFrequentness(): Convert distribution to cumulative probabilities
 *   - findProbabilisticSupport(): Binary search for maximum valid support
 *
 * Subclasses implement distribution computation:
 *   - DirectConvolutionSupportCalculator: Dynamic Programming O(n²)
 *   - FFTConvolutionSupportCalculator: Fast Fourier Transform O(n log² n)
 *
 * Mathematical Foundation:
 *
 * 1. DISTRIBUTION: P(support = s) for each possible support value s
 *    Computed via Generating Function polynomial multiplication
 *
 * 2. FREQUENTNESS: P(support ≥ s) = Σ P(support = i) for i = s to n
 *    Cumulative sum from right to left
 *
 * 3. PROBABILISTIC SUPPORT: max{s : P(support ≥ s) ≥ τ}
 *    Found via binary search on monotonic frequentness array
 *
 * Example:
 *   Distribution: [0.1, 0.3, 0.4, 0.2] (P(sup=0), P(sup=1), P(sup=2), P(sup=3))
 *   Frequentness: [1.0, 0.9, 0.6, 0.2] (cumulative from right)
 *   If τ = 0.7, ProbSupport = 1 (largest s where freq[s] ≥ 0.7)
 *
 * @author Dang Nguyen Le, Gia Huy Vo
 */
public abstract class AbstractSupportCalculator implements SupportCalculator {

    /**
     * Probability threshold τ (tau).
     * Pattern is considered frequent if P(support ≥ s) ≥ τ.
     * Typical values: 0.5 to 0.9
     */
    protected final double tau;


    /**
     * Constructor with parameter validation.
     *
     * @param tau probability threshold, must be in (0, 1]
     * @throws IllegalArgumentException if tau is out of range
     */
    public AbstractSupportCalculator(double tau) {
        // Validate tau is in valid range (0, 1]
        // tau = 0 would accept any support (meaningless)
        // tau > 1 is impossible (probability can't exceed 1)
        if (tau <= 0.0 || tau > 1.0) {
            throw new IllegalArgumentException("Tau must be in (0, 1], got: " + tau);
        }

        this.tau = tau;
    }

    /**
     * Compute frequentness array from probability distribution.
     *
     * Transforms: distribution[s] = P(support = s)
     * Into:       frequentness[s] = P(support ≥ s)
     *
     * Algorithm: Cumulative sum from RIGHT to LEFT
     *   frequentness[n-1] = distribution[n-1]
     *   frequentness[i] = frequentness[i+1] + distribution[i]
     *
     * Why right-to-left?
     *   P(support ≥ s) = P(support = s) + P(support = s+1) + ... + P(support = n)
     *   Starting from right allows single-pass computation
     *
     * Example:
     *   distribution = [0.1, 0.3, 0.4, 0.2]
     *   frequentness[3] = 0.2
     *   frequentness[2] = 0.2 + 0.4 = 0.6
     *   frequentness[1] = 0.6 + 0.3 = 0.9
     *   frequentness[0] = 0.9 + 0.1 = 1.0
     *   Result: [1.0, 0.9, 0.6, 0.2]
     *
     * Property: frequentness is monotonically NON-INCREASING
     *   frequentness[0] ≥ frequentness[1] ≥ ... ≥ frequentness[n-1]
     *   This enables binary search in findProbabilisticSupport()
     *
     * @param distribution probability distribution where distribution[s] = P(support = s)
     * @return frequentness array where frequentness[s] = P(support ≥ s)
     */
    protected double[] computeFrequentness(double[] distribution) {
        int n = distribution.length;

        // Allocate result array
        double[] frequentness = new double[n];

        // Start from rightmost element (highest support)
        // P(support ≥ n-1) = P(support = n-1)
        frequentness[n - 1] = distribution[n - 1];

        // Build cumulative sum from right to left
        // P(support ≥ i) = P(support ≥ i+1) + P(support = i)
        for (int i = n - 2; i >= 0; i--) {
            frequentness[i] = frequentness[i + 1] + distribution[i];
        }

        return frequentness;
    }

    /**
     * Find maximum support s such that P(support ≥ s) ≥ τ.
     *
     * This is the DEFINITION of probabilistic support:
     *   ProbSupport_τ(X) = max{s : P(support(X) ≥ s) ≥ τ}
     *
     * Algorithm: Binary Search
     *   - Frequentness array is monotonically non-increasing
     *   - Find largest index where value ≥ τ
     *
     * Binary Search Logic:
     *   - If frequentness[mid] ≥ τ: valid support, try larger (search right)
     *   - If frequentness[mid] < τ: invalid, try smaller (search left)
     *
     * Example: frequentness = [1.0, 0.9, 0.6, 0.2], τ = 0.7
     *   mid=1: freq[1]=0.9 ≥ 0.7 ✓ → result=1, search right
     *   mid=2: freq[2]=0.6 < 0.7 ✗ → search left
     *   Final: result = 1
     *
     * Floating-point handling:
     *   Uses EPSILON tolerance to handle precision errors
     *   freq[mid] ≥ τ - EPSILON instead of freq[mid] ≥ τ
     *
     *
     * @param frequentness cumulative probability array (monotonically non-increasing)
     * @return maximum support s where P(support ≥ s) ≥ τ
     */
    protected int findProbabilisticSupport(double[] frequentness) {
        // Binary search bounds
        int left = 0;
        int right = frequentness.length - 1;

        // Track best valid support found
        int result = 0;

        // Binary search for largest valid support
        while (left <= right) {
            // Calculate midpoint (avoids overflow for large arrays)
            int mid = left + (right - left) / 2;

            // Check if this support value is valid (probability ≥ τ)
            // Use EPSILON tolerance for floating-point comparison
            if (frequentness[mid] >= tau - Constants.EPSILON) {
                // Valid support found, record it
                result = mid;

                // Try to find larger valid support (search right half)
                left = mid + 1;
            } else {
                // Probability too low, try smaller support (search left half)
                right = mid - 1;
            }
        }

        return result;
    }
}
