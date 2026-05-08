package domain.support;

import domain.model.Tidset;
import shared.Constants;

/**
 * Normal approximation support calculator — used by TopKPFIM (Li et al. 2017).
 *
 * Estimates probabilistic support Λᵖᵣ(X) using the Lyapunov CLT approximation:
 *
 *   P(Λ(X) ≥ λ) ≈ Φ((λ - 0.5 - E(X)) / √S²(X))
 *
 * where:
 *   E(X) = Σ p_i      (mean of Poisson-binomial = expected support)
 *   S²(X) = Σ p_i(1-p_i)   (variance of Poisson-binomial)
 *   Φ = standard normal CDF
 *
 * Probabilistic support: Λᵖᵣ(X) = max{i | P(Λ(X) ≥ i) > τ}
 *
 * Reference: Li et al. 2017, Section 2.4, Equation 3.
 * Reduces cost from O(N² log N) (exact Poisson-binomial) to O(N).
 *
 * Cost: O(|tidset|) per query — the dominant savings vs DirectConvolution
 * which is O(|tidset|²).
 */
public class NormalApproxSupportCalculator implements SupportCalculator {

    private final double tau;

    public NormalApproxSupportCalculator(double tau) {
        if (tau <= 0.0 || tau >= 1.0) throw new IllegalArgumentException("tau must be in (0,1)");
        this.tau = tau;
    }

    @Override
    public int computeProbabilisticSupport(double[] probs, int minRequiredSupport) {
        double[] r = computeProbabilisticSupportWithFrequentness(probs);
        return (int) r[0];
    }

    @Override
    public double computeProbability(double[] probs, int supportLevel) {
        double mean = 0, var = 0;
        for (double p : probs) { mean += p; var += p * (1.0 - p); }
        return tailProb(supportLevel, mean, var);
    }

    @Override
    public String getStrategyName() { return "NormalApprox"; }

    @Override
    public double[] computeProbabilisticSupportWithFrequentness(double[] probs) {
        double mean = 0, var = 0;
        for (double p : probs) { mean += p; var += p * (1.0 - p); }
        return findSupportAndProb(mean, var);
    }

    @Override
    public double[] computeProbabilisticSupportFromTidset(Tidset tidset, int dbSize) {
        double mean = 0, var = 0;
        for (Tidset.TIDProb e : tidset.getEntries()) {
            mean += e.prob;
            var += e.prob * (1.0 - e.prob);
        }
        return findSupportAndProb(mean, var);
    }

    /** Find max i such that P(Λ ≥ i) > τ, using normal CDF approximation. */
    private double[] findSupportAndProb(double mean, double var) {
        if (var < Constants.EPSILON) {
            // Degenerate: all probs are 0 or 1 → support = round(mean), prob = 1
            int sup = (int) Math.round(mean);
            return new double[] { sup, 1.0 };
        }
        // Search for largest i with P(Λ ≥ i) > τ via binary search over [0, ceil(mean)+1]
        int upper = (int) Math.ceil(mean) + 1;
        int sup = 0;
        int left = 0, right = upper;
        while (left <= right) {
            int mid = left + (right - left) / 2;
            double p = tailProb(mid, mean, var);
            if (p > tau) { sup = mid; left = mid + 1; }
            else right = mid - 1;
        }
        double prob = tailProb(sup, mean, var);
        return new double[] { sup, prob };
    }

    /** P(Λ ≥ i) ≈ 1 - Φ((i - 0.5 - mean) / √var) */
    private double tailProb(int i, double mean, double var) {
        if (var < Constants.EPSILON) return (mean >= i) ? 1.0 : 0.0;
        double z = (i - 0.5 - mean) / Math.sqrt(var);
        return 1.0 - normalCdf(z);
    }

    /** Standard normal CDF using Abramowitz-Stegun approximation. */
    private static double normalCdf(double x) {
        if (x < 0) return 1.0 - normalCdf(-x);
        double t = 1.0 / (1.0 + 0.2316419 * x);
        double d = 0.3989422804 * Math.exp(-x * x / 2.0);
        double p = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))));
        return 1.0 - p;
    }
}