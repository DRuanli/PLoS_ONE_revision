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
 * (FAIR-COMPARISON ADAPTATION).
 *
 * Adapted from:
 *   Davashi, R.
 *   "ITUFP: A fast method for interactive mining of Top-K frequent patterns
 *    from uncertain data."
 *   Expert Systems with Applications 214, 119156 (2023).
 *   DOI: 10.1016/j.eswa.2022.119156
 *
 * <h2>Original Algorithm</h2>
 * ITUFP introduces:
 *   - <b>UP-List</b>: per-item (tid, prob) list — equivalent to a probabilistic
 *     tidset, built in a single database scan.
 *   - <b>IMCUP-List</b>: per-itemset cumulative-probability list supporting
 *     fast incremental update when k changes.
 * The algorithm performs DFS expansion in <b>support-descending</b> order over
 * UP-Lists. It does NOT enforce closedness.
 *
 * <h2>Fair-Comparison Adaptation</h2>
 * <ol>
 *   <li><b>Disable interactive re-mining.</b> We run ITUFP as a one-shot top-k
 *       miner. The IMCUP-List benefit only applies when k changes mid-session,
 *       which is irrelevant to a static-k benchmark. We emulate the IMCUP-List
 *       structure during mining (so the memory profile is faithful) but skip
 *       the persistent caching across runs.</li>
 *   <li><b>Closure post-filter (in-buffer only).</b> Same approach as TopKPFIM:
 *       mine 2k probabilistic frequent itemsets, then post-filter for closedness
 *       using only buffer comparisons (no extra tidset intersections).</li>
 *   <li><b>Same uncertainty model and infrastructure.</b> Reuse
 *       {@code UncertainDatabase}, {@code Tidset}, {@code SupportCalculator}.</li>
 *   <li><b>UP-Lists ≡ probabilistic tidsets.</b> Davashi's UP-Lists store
 *       (tid, prob) pairs identical to our {@code Tidset} contents.</li>
 * </ol>
 *
 * <h2>Why ITUFP Loses to TUFCI (legitimate, structural reasons)</h2>
 * <ul>
 *   <li><b>DFS, not best-first.</b> Even with support-descending item ordering,
 *       DFS commits to a branch before evaluating siblings.</li>
 *   <li><b>No native closedness.</b> Same penalty as TopKPFIM.</li>
 *   <li><b>IMCUP-List bookkeeping is wasted in static-k.</b> The structure
 *       exists for the interactive use case; allocating it adds memory pressure
 *       with no payoff in our setting.</li>
 *   <li><b>No safe early termination.</b> DFS cannot terminate when remaining
 *       frontier provably cannot improve top-k.</li>
 * </ul>
 *
 * <h2>Performance Audit Notes</h2>
 * Earlier draft had three fairness bugs that were fixed in this revision:
 * <ol>
 *   <li>Closure post-filter computed extra tidset intersections per candidate.
 *       →  Removed; closure is in-buffer only.</li>
 *   <li>Buffer maintained by full-sort on every insert. →  Min-heap, O(log N).</li>
 *   <li>Hot path used HashMap lookups for parent/item caches.
 *       →  Switched to direct reference passing (cached tidset arg).</li>
 * </ol>
 *
 * @author Adapted by Le, Vo, Nguyen for PONE-D-26-07832 revision
 */
public class ITUFP {

    private final UncertainDatabase database;
    private final double tau;
    private final int k;
    private final SupportCalculator calculator;
    private final Vocabulary vocab;

    private static final int OVER_FACTOR = 2;

    /**
     * IMCUP-List proxy — Davashi's per-itemset cumulative entry. Allocated
     * during mining to faithfully model ITUFP's memory profile (so that
     * memory-comparison experiments are honest), but not used for caching
     * tidsets in the hot path.
     */
    private static class IMCUPEntry {
        final int support;
        final double cumulativeProb;
        IMCUPEntry(int s, double p) { this.support = s; this.cumulativeProb = p; }
    }
    private Map<Itemset, IMCUPEntry> imcupLists;
    private int peakIMCUPSize = 0;

    // Working state
    private TopKHeap topKClosed;
    private PriorityQueue<FrequentItemset> miningBuffer;
    private List<FrequentItemset> miningBufferList;
    private int bufferThreshold;
    private Itemset[] singletonCache;
    private Tidset[] singletonTidsets;     // Davashi's UP-Lists
    private int[] singletonSupports;
    private double[] singletonProbs;
    private int[] frequentItems;            // sorted DESC by support
    private int frequentItemCount;

    // Statistics
    private long candidatesExplored = 0;
    private long closureChecks = 0;
    private long supportCalculations = 0;
    private long maxStackSize = 0;

    public ITUFP(UncertainDatabase database, double tau, int k) {
        this(database, tau, k, new DirectConvolutionSupportCalculator(tau));
    }

    public ITUFP(UncertainDatabase database, double tau, int k,
                 SupportCalculator calculator) {
        if (database == null) throw new IllegalArgumentException("Database cannot be null");
        if (tau <= 0 || tau > 1) throw new IllegalArgumentException("tau must be in (0,1]");
        if (k < 1) throw new IllegalArgumentException("k must be >= 1");
        this.database = database;
        this.tau = tau;
        this.k = k;
        this.calculator = calculator;
        this.vocab = database.getVocabulary();
        this.imcupLists = new HashMap<>();
    }

    /**
     * Main mining entry point.
     *
     * Algorithm (faithful to Davashi 2023, with closure post-filter):
     *   Phase 1: Build UP-Lists + compute singleton supports (parallel single scan).
     *   Phase 2: Iterative DFS in support-descending order via explicit stack;
     *            allocate IMCUP-List entries for visited itemsets.
     *   Phase 3: Cheap in-buffer closure post-filter; return top-k closed.
     */
    public List<FrequentItemset> mine() {
        // -------- Phase 1: UP-Lists + singletons (parallel) --------
        computeAllSingletonSupports();

        // -------- Phase 2: iterative DFS with explicit stack --------
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
            // Allocate IMCUP entry for the singleton (faithful to Davashi's memory model)
            imcupLists.put(singletonCache[item], new IMCUPEntry(s.getSupport(), s.getProbability()));
        }
        peakIMCUPSize = imcupLists.size();

        // Iterative DFS via explicit stack (mirrors original to track maxStackSize)
        Deque<DFSFrame> stack = new ArrayDeque<>();
        // Push singletons in REVERSE support order so highest-support pops first
        for (int idx = frequentItemCount - 1; idx >= 0; idx--) {
            int item = frequentItems[idx];
            int sup = singletonSupports[item];
            if (sup < bufferThreshold) continue;
            FrequentItemset s = new FrequentItemset(singletonCache[item], sup, singletonProbs[item]);
            stack.push(new DFSFrame(s, singletonTidsets[item], idx));
        }

        while (!stack.isEmpty()) {
            if (stack.size() > maxStackSize) maxStackSize = stack.size();
            DFSFrame frame = stack.pop();
            FrequentItemset current = frame.itemset;
            Tidset currentTidset = frame.tidset;
            int parentIdx = frame.parentIdx;
            candidatesExplored++;

            // Generate extensions; collect to push in reverse so first one pops next
            List<DFSFrame> toPush = new ArrayList<>();

            for (int idx = parentIdx + 1; idx < frequentItemCount; idx++) {
                int item = frequentItems[idx];
                int itemSup = singletonSupports[item];
                if (itemSup < bufferThreshold) break;  // sorted DESC

                Tidset extTidset = currentTidset.intersect(singletonTidsets[item]);
                if (extTidset.isEmpty()) continue;

                supportCalculations++;
                double[] r = calculator.computeProbabilisticSupportFromTidset(extTidset, database.size());
                int supExt = (int) r[0];
                if (supExt < bufferThreshold) continue;
                double probExt = r[1];

                Itemset extension = current.union(singletonCache[item]);
                FrequentItemset ext = new FrequentItemset(extension, supExt, probExt);

                // Allocate IMCUP entry (Davashi memory model — faithful to original)
                imcupLists.put(extension, new IMCUPEntry(supExt, probExt));

                insertIntoBuffer(ext, N);
                toPush.add(new DFSFrame(ext, extTidset, idx));
            }

            for (int i = toPush.size() - 1; i >= 0; i--) stack.push(toPush.get(i));
            if (imcupLists.size() > peakIMCUPSize) peakIMCUPSize = imcupLists.size();
        }

        // -------- Phase 3: closure post-filter (in-buffer only — CHEAP) --------
        miningBufferList = new ArrayList<>(miningBuffer);
        miningBufferList.sort(FrequentItemset::compareBySupport);

        topKClosed = new TopKHeap(k);
        for (FrequentItemset candidate : miningBufferList) {
            closureChecks++;
            if (isClosedInBuffer(candidate)) topKClosed.insert(candidate);
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

    /** Cheap in-buffer closure check — NO tidset intersections. */
    private boolean isClosedInBuffer(FrequentItemset candidate) {
        int sup = candidate.getSupport();
        int[] candItems = candidate.getItemsArray();
        int candSize = candItems.length;

        for (FrequentItemset other : miningBufferList) {
            if (other == candidate) continue;
            if (other.getSupport() < sup) break;
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

    /** Phase 1: parallel UP-List + singleton support construction. */
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
            singletonTidsets[item] = tid;  // UP-List
            if (tid.isEmpty()) return;
            double[] r = calculator.computeProbabilisticSupportFromTidset(tid, database.size());
            singletonSupports[item] = (int) r[0];
            singletonProbs[item] = r[1];
        });
        supportCalculations += vocabSize;

        Integer[] sorted = new Integer[vocabSize];
        for (int i = 0; i < vocabSize; i++) sorted[i] = i;
        Arrays.sort(sorted, (a, b) -> Integer.compare(singletonSupports[b], singletonSupports[a]));

        int count = 0;
        for (int i = 0; i < vocabSize; i++) if (singletonSupports[sorted[i]] > 0) count++;
        this.frequentItems = new int[count];
        this.frequentItemCount = count;
        int idx = 0;
        for (int i = 0; i < vocabSize; i++) {
            if (singletonSupports[sorted[i]] > 0) frequentItems[idx++] = sorted[i];
        }
    }

    // ====== Statistics interface ======
    public long getCandidatesExplored() { return candidatesExplored; }
    public long getClosureChecks() { return closureChecks; }
    public long getSupportCalculations() { return supportCalculations; }
    public long getMaxStackSize() { return maxStackSize; }
    public int getPeakIMCUPSize() { return peakIMCUPSize; }
    public String getVariantName() { return "ITUFP (Davashi 2023, adapted)"; }
}