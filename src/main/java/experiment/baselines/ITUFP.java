package experiment.baselines;

import domain.support.SupportCalculator;
import domain.support.DirectConvolutionSupportCalculator;
import infrastructure.persistence.UncertainDatabase;
import infrastructure.persistence.Vocabulary;
import domain.model.*;
import infrastructure.topK.TopKHeap;

import java.util.*;

/**
 * ITUFP — Interactive Top-K Uncertain Frequent Pattern mining
 * (ADAPTED to TUFCI's problem: top-k probabilistic frequent closed itemset mining).
 *
 * Reference:
 *   Davashi, R. "ITUFP: A fast method for interactive mining of Top-K frequent
 *   patterns from uncertain data." ESWA 214, 119156 (2023).
 *
 * <h2>EXACTNESS NOTE — Critical Implementation Detail</h2>
 * In the original paper, ITUFP uses <em>expected support</em> which is additive
 * per-tuple. This means combining IMCUP-Lists by intersection (multiplying
 * per-tuple probabilities) is mathematically valid for expSup.
 *
 * For TUFCI's problem (probabilistic support, requires the full distribution),
 * naive intersection of two derived IMCUP-Lists DOUBLE-COUNTS shared items'
 * probabilities. E.g., combining {A,B} with {A,C} via tidset intersection
 * gives entries with prob = P(A)·P(B) · P(A)·P(C) = P(A)²·P(B)·P(C), which
 * is wrong; the correct combined probability is P(A)·P(B)·P(C).
 *
 * <b>Fix</b>: when extending an IMCUP-List X with one more item y to form
 * X∪{y}, we always intersect X.tidset with the fresh singleton tidset
 * {y}.tidset (which contains only P(y), not products). This yields exact
 * probabilistic support and matches TUFCI's ground truth.
 *
 * The recursive DFS structure of ITUFP_growth still walks the prefix-tree
 * exactly once per itemset; we just look up the correct singleton tidset
 * for each extension rather than reusing a sibling IMCUP-List.
 *
 * <h2>What Is Faithful to the Paper</h2>
 * <ul>
 *   <li>UP-List per item with Max header (Section 3.3.1).</li>
 *   <li>IMCUP-List per itemset with Index1, Index2 (Definition 5).</li>
 *   <li>Items sorted DESC by support (Algorithm Fig. 7 line 2).</li>
 *   <li>PUFP technique: (u_x.support × max_prob(u_l)) > minSup (line 7).</li>
 *   <li>Recursive divide-and-conquer (Mine_Patterns + ITUFP_growth, Fig. 8).</li>
 *   <li>Tᵏ array with raised minSup (Fig. 7 lines 5-8).</li>
 * </ul>
 *
 * <h2>What Is Adapted</h2>
 * <ul>
 *   <li><b>Support definition</b>: probabilistic support Λᵖᵣ instead of expSup,
 *       computed via TUFCI's DirectConvolutionSupportCalculator (same problem).</li>
 *   <li><b>Tidset extension via singleton</b>: extend IMCUP-List(X) with item y
 *       by intersecting X.tidset with the level-1 singleton tidset of y. This
 *       is mathematically equivalent to the paper's operation under expSup,
 *       but exact under probabilistic support.</li>
 *   <li><b>ppf removed</b>: paper's ppf early-termination is sound only for
 *       additive measures. For probabilistic support, truncated tidsets give
 *       wrong supports for descendants. PUFP alone is sound and retained.</li>
 *   <li><b>Closure post-filter</b>: in-buffer subset check on FFIC of size 2k.</li>
 * </ul>
 *
 * @author Adapted by Le, Vo, Nguyen for PONE-D-26-07832 revision
 */
public class ITUFP {

    private final UncertainDatabase database;
    private final double tau;
    private final int k;
    private final SupportCalculator calculator;
    private final Vocabulary vocab;

    private static final int OVER_FACTOR = 3;

    /** UP-List / IMCUP-List unified record. */
    private static class UPList {
        final Itemset itemset;
        final Tidset tidset;
        final int support;        // probabilistic support Λᵖᵣ
        final double prob;
        final double maxProb;     // Max header for PUFP
        /**
         * The largest level-1 index (in {@code level1}) used to construct this
         * itemset. Extensions add only items at indices > lastLevel1Idx, which
         * enforces the prefix-enumeration constraint and ensures each itemset
         * is generated exactly once.
         */
        final int lastLevel1Idx;
        int index1 = -1;
        int index2 = -1;

        UPList(Itemset i, Tidset t, int s, double p, double m, int lastIdx) {
            this.itemset = i; this.tidset = t;
            this.support = s; this.prob = p; this.maxProb = m;
            this.lastLevel1Idx = lastIdx;
        }
    }

    private List<UPList> tk;
    private int minSup;

    private List<UPList> ffic;
    private UPList[] level1;
    private int level1Count;
    private Map<Itemset, UPList> imcupRegistry;

    // Statistics
    private long candidatesExplored = 0;
    private long closureChecks = 0;
    private long supportCalculations = 0;
    private int peakIMCUPSize = 0;
    private int maxRecursionDepth = 0;
    private int currentDepth = 0;

    public ITUFP(UncertainDatabase database, double tau, int k) {
        this(database, tau, k, new DirectConvolutionSupportCalculator(tau));
    }

    public ITUFP(UncertainDatabase database, double tau, int k, SupportCalculator calculator) {
        if (database == null) throw new IllegalArgumentException("Database cannot be null");
        if (tau <= 0 || tau > 1) throw new IllegalArgumentException("tau must be in (0,1]");
        if (k < 1) throw new IllegalArgumentException("k must be >= 1");
        this.database = database;
        this.tau = tau;
        this.k = k;
        this.calculator = calculator;
        this.vocab = database.getVocabulary();
        this.imcupRegistry = new HashMap<>();
    }

    public List<FrequentItemset> mine() {
        buildUPLists();

        int N = OVER_FACTOR * k;
        tk = new ArrayList<>(k);
        ffic = new ArrayList<>(N);
        minSup = 0;

        for (int i = 0; i < level1Count; i++) {
            insertTk(level1[i]);
            insertFFIC(level1[i], N);
        }

        minePatterns();

        // ======== Closure verification (paper-justifiable strict check) ========
        // Original ITUFP does NOT enforce closedness. To produce top-k CLOSED
        // itemsets, each candidate must be verified: X is closed iff no
        // immediate extension X∪{y} (for any y not in X) has Λᵖᵣ(X∪{y}) = Λᵖᵣ(X).
        // This requires an explicit support computation for each (candidate, item)
        // pair — the unavoidable cost of adapting a non-closed miner to the
        // closed-mining problem.
        ffic.sort((a, b) -> Integer.compare(b.support, a.support));

        TopKHeap topKClosed = new TopKHeap(k);
        for (UPList cand : ffic) {
            closureChecks++;
            if (isClosedStrict(cand)) {
                FrequentItemset fi = new FrequentItemset(cand.itemset, cand.support, cand.prob);
                topKClosed.insert(fi);
            }
        }

        List<FrequentItemset> result = topKClosed.getAll();
        result.sort(FrequentItemset::compareBySupport);
        return result;
    }

    /** Sub-procedure Mine_Patterns (paper Fig. 8 lines 1-32). */
    private void minePatterns() {
        for (int xIdx = 0; xIdx < level1Count; xIdx++) {
            UPList ux = level1[xIdx];
            if (ux.support <= minSup) continue;

            // Build all 2-itemsets {x_idx, l_idx} for l_idx > xIdx that pass PUFP
            List<UPList> imc = new ArrayList<>();

            for (int lIdx = xIdx + 1; lIdx < level1Count; lIdx++) {
                UPList ul = level1[lIdx];
                if (ul.support <= minSup) continue;

                // PUFP: anti-monotonicity upper bound
                if ((ux.support * ul.maxProb) <= minSup) continue;

                Itemset combined = ux.itemset.union(ul.itemset);
                UPList c = extendByLevel1(ux, lIdx, combined);
                if (c == null || c.support <= minSup) continue;

                c.index1 = ux.tidset.size() - 1;
                c.index2 = ul.tidset.size() - 1;

                imc.add(c);
                imcupRegistry.put(combined, c);
                if (imcupRegistry.size() > peakIMCUPSize) peakIMCUPSize = imcupRegistry.size();

                insertFFIC(c, OVER_FACTOR * k);
                if (c.support > minSup) insertTk(c);
            }

            if (imc.size() > 1) itufpGrowth(imc);
        }
    }

    /**
     * Sub-procedure ITUFP_growth (paper Fig. 8 lines 33-63).
     *
     * Each candidate cx in imc represents an itemset X = prefix ∪ {item at lastLevel1Idx}.
     * Children of cx are extensions cx ∪ {y} where y is at level-1 index > cx.lastLevel1Idx.
     *
     * IMPORTANT: extension is done by intersecting cx.tidset with level1[yIdx].tidset
     * (the singleton tidset for y), NOT by intersecting two derived IMCUP-Lists.
     * This avoids the probability double-counting bug.
     */
    private void itufpGrowth(List<UPList> imc) {
        currentDepth++;
        if (currentDepth > maxRecursionDepth) maxRecursionDepth = currentDepth;
        candidatesExplored++;

        for (int xIdx = 0; xIdx < imc.size(); xIdx++) {
            UPList cx = imc.get(xIdx);
            if (cx.support <= minSup) continue;

            List<UPList> imcPrime = new ArrayList<>();

            // Extend cx with each level-1 item beyond cx.lastLevel1Idx
            for (int yIdx = cx.lastLevel1Idx + 1; yIdx < level1Count; yIdx++) {
                UPList ly = level1[yIdx];
                if (ly.support <= minSup) continue;

                // PUFP
                if ((cx.support * ly.maxProb) <= minSup) continue;

                Itemset combined = cx.itemset.union(ly.itemset);
                UPList cPrime = extendByLevel1(cx, yIdx, combined);
                if (cPrime == null || cPrime.support <= minSup) continue;

                cPrime.index1 = cx.tidset.size() - 1;
                cPrime.index2 = ly.tidset.size() - 1;

                imcPrime.add(cPrime);
                imcupRegistry.put(combined, cPrime);
                if (imcupRegistry.size() > peakIMCUPSize) peakIMCUPSize = imcupRegistry.size();

                insertFFIC(cPrime, OVER_FACTOR * k);
                if (cPrime.support > minSup) insertTk(cPrime);
            }

            if (imcPrime.size() > 1) itufpGrowth(imcPrime);
        }

        currentDepth--;
    }

    /**
     * Extend an IMCUP-List (or UP-List) X with the level-1 singleton at index newItemLevelIdx.
     * Computes EXACT probabilistic support by intersecting with the singleton tidset
     * (which carries only P(new_item)) — not with another derived IMCUP-List.
     */
    private UPList extendByLevel1(UPList parent, int newItemLevelIdx, Itemset combined) {
        UPList singleton = level1[newItemLevelIdx];

        // Intersect parent's tidset with the SINGLETON tidset (carries only P(newItem))
        Tidset joined = parent.tidset.intersect(singleton.tidset);
        if (joined.isEmpty()) return null;

        // Cheap pre-check: |joined| is loose upper bound on Λᵖᵣ
        if (joined.size() <= minSup) return null;

        // Compute max prob for PUFP at next level
        double maxProbJoined = 0;
        for (Tidset.TIDProb e : joined.getEntries()) {
            if (e.prob > maxProbJoined) maxProbJoined = e.prob;
        }

        supportCalculations++;
        double[] r = calculator.computeProbabilisticSupportFromTidset(joined, database.size());
        int sup = (int) r[0];
        if (sup <= minSup) return null;
        double prob = r[1];

        return new UPList(combined, joined, sup, prob, maxProbJoined, newItemLevelIdx);
    }

    private void insertTk(UPList list) {
        if (tk.size() < k) {
            tk.add(list);
            tk.sort((x, y) -> Integer.compare(y.support, x.support));
            if (tk.size() == k) minSup = tk.get(tk.size() - 1).support;
            return;
        }
        UPList worst = tk.get(tk.size() - 1);
        if (list.support <= worst.support) return;
        tk.set(tk.size() - 1, list);
        tk.sort((x, y) -> Integer.compare(y.support, x.support));
        minSup = tk.get(tk.size() - 1).support;
    }

    private void insertFFIC(UPList list, int N) {
        if (ffic.size() < N) { ffic.add(list); return; }
        int minIdx = 0;
        int minVal = ffic.get(0).support;
        for (int i = 1; i < ffic.size(); i++) {
            if (ffic.get(i).support < minVal) { minVal = ffic.get(i).support; minIdx = i; }
        }
        if (list.support > minVal) ffic.set(minIdx, list);
    }

    /**
     * Strict closure check: X is closed iff for every item y not in X with
     * y's singleton support ≥ Λᵖᵣ(X), the extension X∪{y} has Λᵖᵣ(X∪{y}) < Λᵖᵣ(X).
     *
     * This requires explicit support computation per (candidate, extension) pair.
     * Cost: up to |frequent items| × |FFIC| extra support calls.
     *
     * This is the unavoidable cost of adapting a non-closed top-k miner to the
     * closed-mining problem — TUFCI avoids this by enforcing closedness natively
     * during search via P7 (support-ordered closure exam).
     */
    private boolean isClosedStrict(UPList candidate) {
        int sup = candidate.support;
        // First, fast in-buffer check (covers cases where superset already mined)
        int[] candItems = candidate.itemset.getItemsArray();
        int candSize = candItems.length;
        for (UPList other : ffic) {
            if (other == candidate) continue;
            if (other.support != sup) continue;
            if (other.itemset.size() <= candSize) continue;
            if (containsAll(other.itemset.getItemsArray(), candItems)) return false;
        }

        // Then strict check: try every level-1 item y not in X
        for (int yIdx = 0; yIdx < level1Count; yIdx++) {
            UPList ly = level1[yIdx];
            int yItem = ly.itemset.getItemsArray()[0];
            if (candidate.itemset.contains(yItem)) continue;
            // Anti-monotonicity: if singleton support < sup, extension can't have equal sup
            if (ly.support < sup) continue;

            // Compute Λᵖᵣ(X ∪ {y}) explicitly
            Tidset joined = candidate.tidset.intersect(ly.tidset);
            if (joined.isEmpty()) continue;
            if (joined.size() < sup) continue;  // can't reach sup

            supportCalculations++;
            double[] r = calculator.computeProbabilisticSupportFromTidset(joined, database.size());
            int extSup = (int) r[0];
            if (extSup == sup) return false;  // closure violated
        }
        return true;
    }

    private boolean isClosedInFFIC(UPList candidate) {
        int sup = candidate.support;
        int[] candItems = candidate.itemset.getItemsArray();
        int candSize = candItems.length;

        for (UPList other : ffic) {
            if (other == candidate) continue;
            if (other.support != sup) continue;
            if (other.itemset.size() <= candSize) continue;
            if (containsAll(other.itemset.getItemsArray(), candItems)) return false;
        }
        return true;
    }

    private static boolean containsAll(int[] a, int[] b) {
        int ai = 0;
        for (int bv : b) {
            while (ai < a.length && a[ai] < bv) ai++;
            if (ai >= a.length || a[ai] != bv) return false;
        }
        return true;
    }

    /** Build UP-Lists, sorted DESC by probabilistic support. */
    private void buildUPLists() {
        int vocabSize = vocab.size();
        Itemset[] singletons = new Itemset[vocabSize];
        UPList[] nodes = new UPList[vocabSize];

        for (int i = 0; i < vocabSize; i++) {
            Itemset s = new Itemset(vocab);
            s.add(i);
            singletons[i] = s;
        }

        java.util.stream.IntStream.range(0, vocabSize).parallel().forEach(item -> {
            Tidset tid = database.getTidset(singletons[item]);
            if (tid.isEmpty()) return;

            double maxProb = 0;
            for (Tidset.TIDProb e : tid.getEntries()) {
                if (e.prob > maxProb) maxProb = e.prob;
            }

            double[] r = calculator.computeProbabilisticSupportFromTidset(tid, database.size());
            int sup = (int) r[0];
            double prob = r[1];

            nodes[item] = new UPList(singletons[item], tid, sup, prob, maxProb, -1 /* placeholder */);
        });
        supportCalculations += vocabSize;

        int count = 0;
        for (UPList n : nodes) if (n != null) count++;
        UPList[] tmp = new UPList[count];
        int idx = 0;
        for (int i = 0; i < vocabSize; i++) {
            if (nodes[i] != null) tmp[idx++] = nodes[i];
        }
        // Sort DESC by support
        Arrays.sort(tmp, (a, b) -> Integer.compare(b.support, a.support));

        // Now assign correct lastLevel1Idx based on sorted position
        level1 = new UPList[count];
        for (int i = 0; i < count; i++) {
            UPList orig = tmp[i];
            level1[i] = new UPList(orig.itemset, orig.tidset, orig.support, orig.prob, orig.maxProb, i);
            imcupRegistry.put(level1[i].itemset, level1[i]);
        }
        level1Count = count;
    }

    public long getCandidatesExplored() { return candidatesExplored; }
    public long getClosureChecks() { return closureChecks; }
    public long getSupportCalculations() { return supportCalculations; }
    public long getMaxStackSize() { return maxRecursionDepth; }
    public int getPeakIMCUPSize() { return peakIMCUPSize; }
    public String getVariantName() { return "ITUFP (Davashi 2023, adapted exact)"; }
}