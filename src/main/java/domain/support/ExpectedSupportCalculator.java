package domain.support;

import domain.model.Tidset;

/**
 * Expected support calculator — used by ITUFP (Davashi 2023).
 *
 * expSup(X) = Σ_T Π_{x∈X} P(x, T)
 *
 * For an itemset X with probabilistic tidset (TID, prob_X_in_T) pairs,
 * this is simply the sum of probabilities. Time complexity: O(|tidset|).
 *
 * Reference: Davashi 2023, Definition 3.
 *   expSup(X) = sum over transactions T of P(X, T)
 *   where P(X, T) = Π_{x∈X} P(x, T) is already encoded in the tidset prob.
 *
 * NOTE: ITUFP does not use a probabilistic confidence τ — it ranks itemsets
 * directly by expected support. This calculator returns expSup as the
 * "support" value (rounded to int) and the same expSup as "probability"
 * for interface compatibility with TUFCI's SupportCalculator contract.
 */
public class ExpectedSupportCalculator implements SupportCalculator {

    public ExpectedSupportCalculator() {}

    @Override
    public int computeProbabilisticSupport(double[] probs, int minRequiredSupport) {
        double sum = 0;
        for (double p : probs) sum += p;
        return (int) sum;  // floor for downstream comparison
    }

    @Override
    public double computeProbability(double[] probs, int supportLevel) {
        double sum = 0;
        for (double p : probs) sum += p;
        return sum;
    }

    @Override
    public String getStrategyName() { return "ExpectedSupport"; }

    @Override
    public double[] computeProbabilisticSupportWithFrequentness(double[] probs) {
        double sum = 0;
        for (double p : probs) sum += p;
        return new double[] { sum, sum };  // [expSup as int-friendly, expSup as double]
    }

    /**
     * Expected support directly from a Tidset (avoids creating dense prob array).
     * Returns [expSup_as_double, expSup_as_double]. We return the first slot as
     * the raw double to preserve precision; callers cast to int as needed.
     */
    @Override
    public double[] computeProbabilisticSupportFromTidset(Tidset tidset, int dbSize) {
        double sum = 0;
        for (Tidset.TIDProb e : tidset.getEntries()) sum += e.prob;
        return new double[] { sum, sum };
    }
}