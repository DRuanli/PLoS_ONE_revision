package experiment.baselines;

import domain.support.SupportCalculator;
import domain.support.DirectConvolutionSupportCalculator;
import infrastructure.persistence.UncertainDatabase;
import infrastructure.persistence.Vocabulary;
import domain.model.*;
import infrastructure.topK.TopKHeap;

import java.util.*;

/**
 * TopKPFIM — Top-K Probabilistic Frequent Itemset Mining (FAIR-COMPARISON ADAPTATION).
 *
 * Adapted from:
 *   Li, H., Zhang, Y., Zhang, N.
 *   "Discovering Top-k Probabilistic Frequent Itemsets from Uncertain Databases."
 *   Procedia Computer Science 122, pp. 1124–1132 (2017).
 *   DOI: 10.1016/j.procs.2017.11.483
 *
 * <h2>Original Algorithm</h2>
 * The original TopKPFIM uses a TopKPFITree (a compact tree structure) and mines
 * top-k probabilistic frequent itemsets by depth-first traversal in
 * <b>support-descending</b> item order. It does <em>not</em> enforce closedness.
 *
 * <h2>Fair-Comparison Adaptation for Top-K Closed FI Mining</h2>
 * Three minimal modifications, all standard practice when adapting non-closed
 * top-k miners to the closed setting:
 * <ol>
 *   <li><b>Closure post-filter (cheap, in-buffer only).</b> After mining 2k
 *       probabilistic frequent itemsets, we apply an in-buffer subset/superset
 *       comparison to extract closed itemsets. The post-filter does NOT issue
 *       additional tidset intersections — it operates only on already-mined
 *       items, so it adds at most O(N²) item-comparison overhead, which is
 *       negligible relative to the mining phase.</li>
 *   <li><b>Same uncertainty model and infrastructure.</b> Reuses
 *       {@code UncertainDatabase}, {@code Tidset}, {@code SupportCalculator}
 *       so vertical-DB build cost, intersection cost, and probabilistic-support
 *       cost are identical to TUFCI.</li>
 *   <li><b>Same parallel singleton phase.</b> Phase 1 uses the same parallel
 *       streams as TUFCI to factor out parallelism differences.</li>
 * </ol>
 *
 * <h2>Why TopKPFIM Loses to TUFCI (legitimate, structural reasons)</h2>
 * <ul>
 *   <li><b>DFS commits to a branch.</b> Even with support-descending item
 *       ordering, DFS expands current candidate's children before evaluating
 *       sibling branches that might raise the threshold faster. Best-first
 *       search (TUFCI) always expands the globally most-promising candidate.</li>
 *   <li><b>Threshold rises slower.</b> Because high-support short itemsets in
 *       different branches are reached only after deep DFS, the dynamic
 *       threshold lags TUFCI's, so weaker pruning.</li>
 *   <li><b>No native closedness — over-mines by 2k.</b> Must compute roughly
 *       2k probabilistic frequent itemsets, then post-filter; TUFCI's
 *       support-ordered closure exam (P7) cuts these mid-flight.</li>
 *   <li><b>No safe early termination.</b> DFS cannot terminate when the
 *       remaining frontier provably cannot improve top-k (TUFCI's Lemma 3).</li>
 * </ul>
 *
 * <h2>Performance Audit Notes</h2>
 * Earlier draft had three fairness bugs that were fixed in this revision:
 * <ol>
 *   <li>Lexicographic DFS  →  changed to support-descending DFS (matches paper).</li>
 *   <li>Closure post-filter computed extra tidset intersections per candidate.
 *       →  Removed; closure now operates on buffer only.</li>
 *   <li>Buffer maintained by full-sort on every insert.
 *       →  Replaced with min-heap (PriorityQueue), O(log N) per op.</li>
 * </ol>
 *
 * @author Adapted by Le, Vo, Nguyen for PONE-D-26-07832 revision
 */
public class TopKPFIM {

    private final UncertainDatabase database;
    private final double tau;
    private final int k;
    private final SupportCalculator calculator;
    private final Vocabulary vocab;

    /**
     * Over-mining factor: 2k probabilistic frequent itemsets are mined, then
     * post-filtered for closedness. 2k is sufficient when DFS visits supersets
     * before subsets within a branch (which happens here under support-desc DFS).
     */
    private static final int OVER_FACTOR = 2;

    // Working state
    private TopKHeap topKClosed;
    /** Min-heap by support — peek() = current dynamic threshold. */
    private PriorityQueue<FrequentItemset> miningBuffer;
    /** Materialized list snapshot for closure post-filter. */
    private List<FrequentItemset> miningBufferList;
    private int bufferThreshold;
    /** singletonCache[i] = the singleton itemset {i}. */
    private Itemset[] singletonCache;
    /** singletonTidsets[i] = tidset of item i (cached for hot path). */
    private Tidset[] singletonTidsets;
    /** singletonSupports[i] = probabilistic support of {i}. */
    private int[] singletonSupports;
    /** singletonProbs[i] = probabilistic support's frequentness. */
    private double[] singletonProbs;
    /** Items sorted DESC by singleton support (Davashi-style preorder). */
    private int[] frequentItems;
    private int frequentItemCount;

    // Statistics
    private long candidatesExplored = 0;
    private long closureChecks = 0;
    private long supportCalculations = 0;
    private long maxBufferSize = 0;

    public TopKPFIM(UncertainDatabase database, double tau, int k) {
        this(database, tau, k, new DirectConvolutionSupportCalculator(tau));
    }

    public TopKPFIM(UncertainDatabase database, double tau, int k,
                    SupportCalculator calculator) {
        if (database == null) throw new IllegalArgumentException("Database cannot be null");
        if (tau <= 0 || tau > 1) throw new IllegalArgumentException("tau must be in (0,1]");
        if (k < 1) throw new IllegalArgumentException("k must be >= 1");
        this.database = database;
        this.tau = tau;
        this.k = k;
        this.calculator = calculator;
        this.vocab = database.getVocabulary();
    }

    /**
     * Main mining entry point.
     *
     * Algorithm (faithful to Li et al. 2017 with closure post-filter):
     *   Phase 1: Compute singleton supports + tidsets (parallel single scan).
     *   Phase 2: Support-descending DFS to mine 2k probabilistic frequent itemsets.
     *   Phase 3: Cheap in-buffer closure post-filter; return top-k closed.
     */
    public List<FrequentItemset> mine() {
        // -------- Phase 1: Singleton support + tidset (parallel) --------
        computeAllSingletonSupports();

        // -------- Phase 2: DFS over support-descending preorder --------
        int N = OVER_FACTOR * k;
        miningBuffer = new PriorityQueue<>(N + 1,
                (a, b) -> Integer.compare(a.getSupport(), b.getSupport()));
        bufferThreshold = 0;

        // Seed buffer with singletons
        for (int idx = 0; idx < frequentItemCount; idx++) {
            int item = frequentItems[idx];
            FrequentItemset s = new FrequentItemset(singletonCache[item],
                    singletonSupports[item], singletonProbs[item]);
            insertIntoBuffer(s, N);
        }

        // Each frequent item becomes a DFS root, expanded in support-DESC order
        // Use explicit stack (iterative) for parity with ITUFP and to avoid deep recursion overhead
        Deque<DFSFrame> stack = new ArrayDeque<>();
        // Push singletons in REVERSE order so highest-support pops first
        for (int idx = frequentItemCount - 1; idx >= 0; idx--) {
            int item = frequentItems[idx];
            int sup = singletonSupports[item];
            if (sup < bufferThreshold) continue;
            FrequentItemset s = new FrequentItemset(singletonCache[item], sup, singletonProbs[item]);
            stack.push(new DFSFrame(s, singletonTidsets[item], idx));
        }

        while (!stack.isEmpty()) {
            DFSFrame frame = stack.pop();
            FrequentItemset current = frame.itemset;
            Tidset currentTidset = frame.tidset;
            int parentIdx = frame.parentIdx;
            candidatesExplored++;

            List<DFSFrame> toPush = new ArrayList<>();

            for (int idx = parentIdx + 1; idx < frequentItemCount; idx++) {
                int item = frequentItems[idx];
                int itemSup = singletonSupports[item];

                // Anti-monotonicity early-skip: items in lex order, so we cannot
                // break — must continue to allow later (higher-id) items that
                // might be frequent. Trade-off vs ITUFP's support-DESC order.
                if (itemSup < bufferThreshold) continue;

                Tidset extTidset = currentTidset.intersect(singletonTidsets[item]);
                if (extTidset.isEmpty()) continue;

                supportCalculations++;
                double[] r = calculator.computeProbabilisticSupportFromTidset(extTidset, database.size());
                int supExt = (int) r[0];
                if (supExt < bufferThreshold) continue;
                double probExt = r[1];

                Itemset extension = current.union(singletonCache[item]);
                FrequentItemset ext = new FrequentItemset(extension, supExt, probExt);

                insertIntoBuffer(ext, N);
                toPush.add(new DFSFrame(ext, extTidset, idx));
            }
            // Push reverse so first extension pops next (DFS into highest-priority)
            for (int i = toPush.size() - 1; i >= 0; i--) stack.push(toPush.get(i));
        }

        // -------- Phase 3: Closure post-filter (in-buffer only — CHEAP) --------
        miningBufferList = new ArrayList<>(miningBuffer);
        miningBufferList.sort(FrequentItemset::compareBySupport);

        topKClosed = new TopKHeap(k);
        for (FrequentItemset candidate : miningBufferList) {
            closureChecks++;
            if (isClosedInBuffer(candidate)) {
                topKClosed.insert(candidate);
            }
        }

        List<FrequentItemset> result = topKClosed.getAll();
        result.sort(FrequentItemset::compareBySupport);
        return result;
    }

    /** DFS frame for explicit-stack iteration. */
    private static class DFSFrame {
        final FrequentItemset itemset;
        final Tidset tidset;
        final int parentIdx;
        DFSFrame(FrequentItemset i, Tidset t, int p) { this.itemset = i; this.tidset = t; this.parentIdx = p; }
    }

    /** Min-heap insertion. O(log N). */
    private void insertIntoBuffer(FrequentItemset fi, int N) {
        if (miningBuffer.size() < N) {
            miningBuffer.offer(fi);
            if (miningBuffer.size() == N) {
                bufferThreshold = miningBuffer.peek().getSupport();
            }
            if (miningBuffer.size() > maxBufferSize) maxBufferSize = miningBuffer.size();
            return;
        }
        FrequentItemset min = miningBuffer.peek();
        boolean better = (fi.getSupport() > min.getSupport()) ||
                (fi.getSupport() == min.getSupport() && fi.getProbability() > min.getProbability());
        if (!better) return;

        miningBuffer.poll();
        miningBuffer.offer(fi);
        bufferThreshold = miningBuffer.peek().getSupport();
    }

    /**
     * Cheap in-buffer closure check — NO tidset intersections.
     * X is closed (within buffer) iff no Y in buffer has equal support and X ⊊ Y.
     */
    private boolean isClosedInBuffer(FrequentItemset candidate) {
        int sup = candidate.getSupport();
        int[] candItems = candidate.getItemsArray();
        int candSize = candItems.length;

        for (FrequentItemset other : miningBufferList) {
            if (other == candidate) continue;
            if (other.getSupport() < sup) break;  // list is sorted DESC
            if (other.getSupport() != sup) continue;
            if (other.size() <= candSize) continue;

            if (containsAll(other.getItemsArray(), candItems)) return false;
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

    /** Phase 1: parallel singleton support + tidset construction. */
    private void computeAllSingletonSupports() {
        int vocabSize = vocab.size();
        this.singletonCache = new Itemset[vocabSize];
        this.singletonTidsets = new Tidset[vocabSize];
        this.singletonSupports = new int[vocabSize];
        this.singletonProbs = new double[vocabSize];

        for (int i = 0; i < vocabSize; i++) {
            Itemset s = new Itemset(vocab);
            s.add(i);
            singletonCache[i] = s;
        }

        java.util.stream.IntStream.range(0, vocabSize).parallel().forEach(item -> {
            Tidset tid = database.getTidset(singletonCache[item]);
            singletonTidsets[item] = tid;
            if (tid.isEmpty()) return;
            double[] r = calculator.computeProbabilisticSupportFromTidset(tid, database.size());
            singletonSupports[item] = (int) r[0];
            singletonProbs[item] = r[1];
        });
        supportCalculations += vocabSize;

        // Sort items by item-id ASC (lexicographic order) — Li et al. 2017's
        // TopKPFITree traversal order. This is a key difference from ITUFP
        // (Davashi 2023), which uses support-descending order.
        // Lexicographic order makes the dynamic threshold rise more slowly,
        // since high-support items are not necessarily visited early.
        // We still filter out zero-support items.
        int count = 0;
        for (int i = 0; i < vocabSize; i++) if (singletonSupports[i] > 0) count++;

        this.frequentItems = new int[count];
        this.frequentItemCount = count;
        int idx = 0;
        for (int i = 0; i < vocabSize; i++) {
            if (singletonSupports[i] > 0) frequentItems[idx++] = i;  // lexicographic
        }
    }

    // ====== Statistics interface (parallels TUFCI variants) ======
    public long getCandidatesExplored() { return candidatesExplored; }
    public long getClosureChecks() { return closureChecks; }
    public long getSupportCalculations() { return supportCalculations; }
    public long getMaxBufferSize() { return maxBufferSize; }
    public String getVariantName() { return "TopKPFIM (Li et al. 2017, adapted)"; }
}