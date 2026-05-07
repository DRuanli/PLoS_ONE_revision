package experiment;

import infrastructure.persistence.UncertainDatabase;
import infrastructure.persistence.Vocabulary;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;

/**
 * UncertaintyInjector — Loads a deterministic dataset from disk and injects
 * uncertainty using the three-stage model described in Section "Dataset
 * Synthesis and Uncertainty Generation":
 *
 *   1. Support-correlated probability:
 *        P_base(i) = P_min + (P_max - P_min) * f(i)^alpha
 *   2. Transaction-length adaptation:
 *        P_final(i, T) = P_base(i) * (1 - 0.5 * max(0, |T| - mu_len) / mu_len)
 *   3. Noise injection:
 *        With probability rho, replace P_final with random in [0.05, P_base].
 *
 * This class exposes all four parameters (P_min, P_max, alpha, rho) so that
 * Exp3 can vary them in sensitivity analysis.
 *
 * Default reference parameters (matching paper):
 *   P_min = 0.1, P_max = 1.0, alpha = 1.0, rho = 0.02
 *
 * Random seed is fixed at 42 for reproducibility unless overridden.
 *
 * @author Le, Vo, Nguyen
 */
public class UncertaintyInjector {

    public static final double DEFAULT_PMIN = 0.1;
    public static final double DEFAULT_PMAX = 1.0;
    public static final double DEFAULT_ALPHA = 1.0;
    public static final double DEFAULT_RHO = 0.02;
    public static final long DEFAULT_SEED = 42L;

    /**
     * Load a deterministic dataset and inject uncertainty using the supplied
     * parameter set.
     *
     * @param filename path to whitespace-delimited dataset file
     *                 Format: "<tid> <item1> <item2> ..."
     * @param pMin minimum base probability
     * @param pMax maximum base probability
     * @param alpha skew exponent for support-correlated probability
     * @param rho noise injection probability
     * @param seed RNG seed
     * @return uncertain database with injected uncertainty
     */
    public static UncertainDatabase load(String filename,
                                          double pMin, double pMax,
                                          double alpha, double rho,
                                          long seed) throws IOException {
        // --- First pass: tally item frequencies and average transaction length ---
        Map<String, Integer> itemFreq = new HashMap<>();
        long totalLen = 0;
        long numTrans = 0;
        List<List<String>> transactions = new ArrayList<>();
        List<Integer> tids = new ArrayList<>();

        try (BufferedReader br = new BufferedReader(new FileReader(filename))) {
            String line;
            while ((line = br.readLine()) != null) {
                line = line.trim();
                if (line.isEmpty()) continue;
                String[] parts = line.split("\\s+");
                if (parts.length < 2) continue;

                int tid;
                try { tid = Integer.parseInt(parts[0]); }
                catch (NumberFormatException e) { continue; }

                List<String> items = new ArrayList<>();
                for (int i = 1; i < parts.length; i++) {
                    items.add(parts[i]);
                    itemFreq.merge(parts[i], 1, Integer::sum);
                }
                if (items.isEmpty()) continue;

                tids.add(tid);
                transactions.add(items);
                totalLen += items.size();
                numTrans++;
            }
        }

        if (numTrans == 0) throw new IOException("Empty dataset: " + filename);
        double muLen = totalLen / (double) numTrans;

        // --- Compute normalized frequency f(i) ---
        int maxFreq = 0;
        for (int v : itemFreq.values()) if (v > maxFreq) maxFreq = v;
        Map<String, Double> normFreq = new HashMap<>();
        for (Map.Entry<String, Integer> e : itemFreq.entrySet()) {
            normFreq.put(e.getKey(), e.getValue() / (double) maxFreq);
        }

        // --- Stage 1: P_base(i) = P_min + (P_max - P_min) * f(i)^alpha ---
        Map<String, Double> pBase = new HashMap<>();
        for (Map.Entry<String, Double> e : normFreq.entrySet()) {
            double f = e.getValue();
            double pb = pMin + (pMax - pMin) * Math.pow(f, alpha);
            pBase.put(e.getKey(), pb);
        }

        // --- Build uncertain database ---
        Vocabulary vocab = new Vocabulary();
        UncertainDatabase db = new UncertainDatabase(vocab);
        Random rng = new Random(seed);

        for (int t = 0; t < transactions.size(); t++) {
            List<String> items = transactions.get(t);
            int tid = tids.get(t);
            int len = items.size();

            // Stage 2: length penalty
            double lengthFactor = 1.0;
            if (len > muLen) {
                lengthFactor = 1.0 - 0.5 * (len - muLen) / muLen;
                if (lengthFactor < 0.0) lengthFactor = 0.0;
            }

            Map<Integer, Double> txn = new HashMap<>();
            for (String name : items) {
                double pb = pBase.get(name);
                double pFinal = pb * lengthFactor;

                // Stage 3: noise injection
                if (rng.nextDouble() < rho) {
                    double noiseLow = 0.05;
                    double noiseHigh = pb;
                    if (noiseHigh < noiseLow) noiseHigh = noiseLow;
                    pFinal = noiseLow + (noiseHigh - noiseLow) * rng.nextDouble();
                }

                if (pFinal <= 0.0) pFinal = 0.001;
                if (pFinal > 1.0) pFinal = 1.0;

                int idx = vocab.getOrCreateIndex(name);
                txn.put(idx, pFinal);
            }
            db.addTransaction(tid, txn);
        }

        db.buildVerticalDatabase();
        db.setName(extractDatasetName(filename));
        return db;
    }

    /** Convenience: load with default parameters (matches paper's reference config). */
    public static UncertainDatabase loadDefault(String filename) throws IOException {
        return load(filename, DEFAULT_PMIN, DEFAULT_PMAX, DEFAULT_ALPHA, DEFAULT_RHO, DEFAULT_SEED);
    }

    private static String extractDatasetName(String filename) {
        String name = filename;
        int slash = Math.max(name.lastIndexOf('/'), name.lastIndexOf('\\'));
        if (slash >= 0) name = name.substring(slash + 1);
        int dot = name.lastIndexOf('.');
        if (dot > 0) name = name.substring(0, dot);
        return name.replace("_uncertain", "");
    }
}