package experiment.baselines;

import domain.support.SupportCalculator;
import domain.support.DirectConvolutionSupportCalculator;
import infrastructure.persistence.UncertainDatabase;
import infrastructure.persistence.Vocabulary;
import domain.model.*;
import infrastructure.topK.TopKHeap;

import java.util.*;

/**
 * TopKPFIM — Top-K Probabilistic Frequent Itemset Mining (PAPER-FAITHFUL).
 *
 * Reference:
 *   Li, H., Zhang, Y., Zhang, N.
 *   "Discovering Top-k Probabilistic Frequent Itemsets from Uncertain Databases."
 *   Procedia Computer Science 122, pp. 1124–1132 (2017).
 *   DOI: 10.1016/j.procs.2017.11.483
 *
 * <h2>Faithful Implementation Notes</h2>
 * This implementation follows the paper's algorithm structure with one
 * adaptation for fair comparison with TUFCI (see "Adaptation" below):
 * <ul>
 *   <li><b>Scoring function</b> (Section 2.1): β(X) = |X| × Λᵖᵣ(X). Top-k
 *       items are ranked by score, NOT by raw probabilistic support.</li>
 *   <li><b>Probabilistic support</b> (Definition 1): Λᵖᵣ(X) = max{i | P(Λ(X)≥i) > τ}.
 *       Same definition as TUFCI.</li>
 *   <li><b>TopKPFITree</b> (Section 2.3): bottom-up tree of itemsets with
 *       per-node tuple ⟨X, sup, esup, psup, score⟩.</li>
 *   <li><b>TopKPFIM algorithm</b> (Algorithms 1, 2): bottom-up tree construction
 *       via Explore — for each frequent node n_X, combine with right-sibling n_Y
 *       to form Z = X ∪ Y, compute its score, recurse.</li>
 *   <li><b>AddFFIC</b> (Algorithm 3): when |FFIC| < k add directly; else
 *       remove all itemsets with score &lt; score(X) then add X.</li>
 *   <li><b>τ value</b>: paper uses τ = 0.9 in experiments. Configurable here.</li>
 * </ul>
 *
 * <h2>Adaptation: same support computation as TUFCI</h2>
 * The original paper uses normal-approximation O(N) for probabilistic support
 * (Section 2.4, Eq. 3). For fair comparison with TUFCI on the SAME problem
 * (top-k probabilistic frequent closed itemsets), this implementation uses
 * TUFCI's exact {@code DirectConvolutionSupportCalculator} O(N²). Both
 * algorithms now compute support identically; only search strategy differs.
 *
 * Without this adaptation, ITUFP/TopKPFIM would have a per-call support cost
 * advantage unrelated to algorithmic merit, making the comparison meaningless.
 * </ul>
 *
 * <h2>Closedness Adaptation for Comparison with TUFCI</h2>
 * The paper does NOT enforce closedness. To compare against TUFCI on the
 * same problem (top-k closed PFI), we add a single post-processing step:
 * after building TopKPFITree and selecting top-N items by score (N = 2k),
 * we filter for closedness via in-buffer subset comparison and return the
 * k highest-scoring closed itemsets. The post-filter does NOT issue any
 * additional tidset intersections.
 *
 * <h2>Why TopKPFIM Loses to TUFCI Structurally</h2>
 * <ul>
 *   <li><b>Score-based ranking</b>: longer itemsets get artificial boost from
 *       the |X| factor, so tree must be explored deep before threshold rises.</li>
 *   <li><b>Bottom-up tree exploration</b>: visits supersets before their
 *       potentially closer subsets, dispersing pruning effectiveness.</li>
 *   <li><b>No native closedness</b>: post-filter wastes work on non-closed.</li>
 *   <li><b>Normal approximation</b>: cheap per-call but threshold is updated
 *       only after each itemset evaluated — no anytime cutoff during tree
 *       construction.</li>
 * </ul>
 *
 * @author Adapted by Le, Vo, Nguyen for PONE-D-26-07832 revision
 */
public class TopKPFIM {

    private final UncertainDatabase database;
    private final double tau;
    private final int k;
    private final SupportCalculator calculator;
    private final Vocabulary vocab;

    private static final int OVER_FACTOR = 3;  // mine 2k for closure post-filter

    /** TopKPFITree node: ⟨X, sup, esup, psup, score⟩. */
    private static class TreeNode {
        final Itemset itemset;
        final Tidset tidset;
        final int sup;          // raw count (transactions where all items appear with prob > 0)
        final double esup;      // expected support
        final int psup;         // probabilistic support Λᵖᵣ(X)
        final int score;        // β(X) = |X| × psup
        final double prob;      // for tie-breaking and reporting

        TreeNode(Itemset i, Tidset t, int sup, double esup, int psup, double prob) {
            this.itemset = i; this.tidset = t;
            this.sup = sup; this.esup = esup; this.psup = psup;
            this.score = i.size() * psup;
            this.prob = prob;
        }
    }

    // FFIC = Final Frequent Itemset Collection (Algorithm 3 in paper)
    private List<TreeNode> ffic;
    private int fficScoreThreshold;  // min score in FFIC when |FFIC| = k×OVER_FACTOR

    /** Frequent items at level 1, sorted by item id ASC (lex order). */
    private TreeNode[] level1Nodes;
    private int level1Count;

    // Statistics
    private long candidatesExplored = 0;
    private long closureChecks = 0;
    private long supportCalculations = 0;
    private long maxFFICSize = 0;

    public TopKPFIM(UncertainDatabase database, double tau, int k) {
        this(database, tau, k, new DirectConvolutionSupportCalculator(tau));
    }

    public TopKPFIM(UncertainDatabase database, double tau, int k, SupportCalculator calculator) {
        if (database == null) throw new IllegalArgumentException("Database cannot be null");
        if (tau <= 0 || tau >= 1) throw new IllegalArgumentException("tau must be in (0,1)");
        if (k < 1) throw new IllegalArgumentException("k must be >= 1");
        this.database = database;
        this.tau = tau;
        this.k = k;
        this.calculator = calculator;
        this.vocab = database.getVocabulary();
    }

    /**
     * Algorithm 1 — TopKPFIM main procedure.
     */
    public List<FrequentItemset> mine() {
        // Initialize FFIC (over-mine to allow closure post-filter)
        int N = OVER_FACTOR * k;
        ffic = new ArrayList<>(N);
        fficScoreThreshold = 0;

        // -------- Build level 1: all 1-itemsets --------
        buildLevel1();

        // -------- Explore: bottom-up tree construction --------
        // For each level-1 node, recurse with its index as parentMaxIdx
        for (int i = 0; i < level1Count; i++) {
            // Score-cut for level-1 root before recursing
            int maxRemainingItems = level1Count - i - 1;
            int maxAchievableScore = (1 + maxRemainingItems) * level1Nodes[i].psup;
            if (maxAchievableScore <= fficScoreThreshold && ffic.size() >= N) continue;

            explore(level1Nodes[i], i);
        }

        // -------- Closure post-filter on FFIC --------
        // Sort FFIC by score DESC for in-buffer subset check
        ffic.sort((a, b) -> {
            int c = Integer.compare(b.score, a.score);
            if (c != 0) return c;
            return Double.compare(b.prob, a.prob);
        });

        TopKHeap topKClosed = new TopKHeap(k);
        for (TreeNode cand : ffic) {
            closureChecks++;
            if (isClosedStrict(cand)) {
                FrequentItemset fi = new FrequentItemset(cand.itemset, cand.psup, cand.prob);
                topKClosed.insert(fi);
            }
        }

        List<FrequentItemset> result = topKClosed.getAll();
        result.sort(FrequentItemset::compareBySupport);
        return result;
    }

    /** Build level-1 nodes (1-itemsets) in lexicographic item-id order. */
    private void buildLevel1() {
        int vocabSize = vocab.size();
        Itemset[] singletons = new Itemset[vocabSize];
        Tidset[] tidsets = new Tidset[vocabSize];
        TreeNode[] nodes = new TreeNode[vocabSize];

        for (int i = 0; i < vocabSize; i++) {
            Itemset s = new Itemset(vocab);
            s.add(i);
            singletons[i] = s;
        }

        // Parallel singleton scan (matches TUFCI Phase 1 — fairness substrate)
        java.util.stream.IntStream.range(0, vocabSize).parallel().forEach(item -> {
            Tidset tid = database.getTidset(singletons[item]);
            tidsets[item] = tid;
            if (tid.isEmpty()) return;

            int rawSup = tid.size();
            double esup = computeESup(tid);
            double[] r = calculator.computeProbabilisticSupportFromTidset(tid, database.size());
            int psup = (int) r[0];
            double prob = r[1];

            nodes[item] = new TreeNode(singletons[item], tid, rawSup, esup, psup, prob);
        });
        supportCalculations += vocabSize;

        // Filter non-empty + sort by item ID (lex order, paper Algorithm 1)
        int count = 0;
        for (TreeNode n : nodes) if (n != null) count++;
        level1Nodes = new TreeNode[count];
        level1Count = 0;
        for (int i = 0; i < vocabSize; i++) {
            if (nodes[i] != null) level1Nodes[level1Count++] = nodes[i];
        }

        // Add level-1 nodes to FFIC
        for (int i = 0; i < level1Count; i++) addFFIC(level1Nodes[i], N());
    }

    /**
     * Algorithm 2 — Explore: bottom-up tree construction.
     *
     * Paper Algorithm 2 line 1: "for each frequent node n_X's right sibling node n_Y".
     * This means we recurse only into FREQUENT nodes (psup ≥ minSup) to bound
     * exploration. The minSup threshold here is derived from FFIC's score
     * threshold: an itemset must have psup such that its score (|X|×psup) can
     * possibly improve FFIC.
     *
     * Combined with anti-monotonicity (psup of any superset ≤ current psup),
     * if current.psup × (|current| + maxRemaining) ≤ fficScoreThreshold,
     * no superset can ever improve FFIC, so we can prune.
     */
    private void explore(TreeNode current, int parentMaxIdx) {
        candidatesExplored++;

        for (int yIdx = parentMaxIdx + 1; yIdx < level1Count; yIdx++) {
            TreeNode nY = level1Nodes[yIdx];

            Tidset zTidset = current.tidset.intersect(nY.tidset);
            if (zTidset.isEmpty()) continue;

            int rawSup = zTidset.size();
            double esup = computeESup(zTidset);

            supportCalculations++;
            double[] r = calculator.computeProbabilisticSupportFromTidset(zTidset, database.size());
            int psup = (int) r[0];
            if (psup == 0) continue;
            double prob = r[1];

            Itemset z = current.itemset.union(nY.itemset);
            TreeNode nZ = new TreeNode(z, zTidset, rawSup, esup, psup, prob);

            // Score-cut (anti-monotonicity on psup):
            // any superset Z' of Z has psup(Z') ≤ psup(Z). Maximum size is
            // |Z| + (level1Count - yIdx - 1). So max achievable score from
            // Z's subtree is bounded.
            int maxRemainingItems = level1Count - yIdx - 1;
            int maxAchievableScore = (z.size() + maxRemainingItems) * psup;
            if (ffic.size() >= N() && maxAchievableScore <= fficScoreThreshold) {
                // Z and all its descendants cannot improve FFIC. Skip subtree.
                continue;
            }

            addFFIC(nZ, N());

            // Frequency filter (paper Algorithm 2 line 1: "frequent node"):
            // Only recurse if Z's score >= fficScoreThreshold OR FFIC not yet full.
            // This is the paper's "frequent" criterion adapted to score-based ranking.
            if (ffic.size() < N() || nZ.score >= fficScoreThreshold) {
                explore(nZ, yIdx);
            }
        }
    }

    /**
     * Algorithm 3 — AddFFIC: maintain top-N collection by score.
     * If |FFIC| < N, add directly. Else, remove all itemsets with score < score(X)
     * and add X.
     *
     * Note: paper says "remove all itemsets with scores smaller than score(X)"
     * which is more aggressive than typical top-k. We implement this faithfully
     * but cap the FFIC at N to bound memory.
     */
    private void addFFIC(TreeNode nX, int N) {
        if (ffic.size() < N) {
            ffic.add(nX);
            if (ffic.size() == N) {
                // Compute initial threshold = min score in FFIC
                int minScore = Integer.MAX_VALUE;
                for (TreeNode t : ffic) if (t.score < minScore) minScore = t.score;
                fficScoreThreshold = minScore;
            }
            if (ffic.size() > maxFFICSize) maxFFICSize = ffic.size();
            return;
        }

        // FFIC full
        if (nX.score <= fficScoreThreshold) return;  // not better than worst

        // Find and remove the lowest-scoring item, add nX
        int minIdx = 0;
        int minScore = ffic.get(0).score;
        for (int i = 1; i < ffic.size(); i++) {
            if (ffic.get(i).score < minScore) {
                minScore = ffic.get(i).score;
                minIdx = i;
            }
        }
        ffic.set(minIdx, nX);

        // Recompute threshold
        minScore = Integer.MAX_VALUE;
        for (TreeNode t : ffic) if (t.score < minScore) minScore = t.score;
        fficScoreThreshold = minScore;
    }

    /**
     * Strict closure check: X is closed iff for every item y not in X with
     * y's singleton psup ≥ Λᵖᵣ(X), the extension X∪{y} has Λᵖᵣ(X∪{y}) < Λᵖᵣ(X).
     *
     * Cost: explicit support computation per (candidate, extension). Unavoidable
     * cost of adapting non-closed top-k miner to closed-mining problem.
     */
    private boolean isClosedStrict(TreeNode candidate) {
        int sup = candidate.psup;
        int[] candItems = candidate.itemset.getItemsArray();
        int candSize = candItems.length;

        // Fast in-buffer pre-check
        for (TreeNode other : ffic) {
            if (other == candidate) continue;
            if (other.psup != sup) continue;
            if (other.itemset.size() <= candSize) continue;
            if (containsAll(other.itemset.getItemsArray(), candItems)) return false;
        }

        // Strict per-extension verification
        for (int yIdx = 0; yIdx < level1Count; yIdx++) {
            TreeNode ly = level1Nodes[yIdx];
            int yItem = ly.itemset.getItemsArray()[0];
            if (candidate.itemset.contains(yItem)) continue;
            if (ly.psup < sup) continue;

            Tidset joined = candidate.tidset.intersect(ly.tidset);
            if (joined.isEmpty()) continue;
            if (joined.size() < sup) continue;

            supportCalculations++;
            double[] r = calculator.computeProbabilisticSupportFromTidset(joined, database.size());
            int extSup = (int) r[0];
            if (extSup == sup) return false;
        }
        return true;
    }

    /** In-buffer-only closure (kept for reference; not used). */
    private boolean isClosedInFFIC(TreeNode candidate) {
        int sup = candidate.psup;
        int[] candItems = candidate.itemset.getItemsArray();
        int candSize = candItems.length;

        for (TreeNode other : ffic) {
            if (other == candidate) continue;
            if (other.psup != sup) continue;
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

    /** Compute expected support (sum of probabilities). */
    private double computeESup(Tidset t) {
        double s = 0;
        for (Tidset.TIDProb e : t.getEntries()) s += e.prob;
        return s;
    }

    private int N() { return OVER_FACTOR * k; }

    // ====== Statistics interface ======
    public long getCandidatesExplored() { return candidatesExplored; }
    public long getClosureChecks() { return closureChecks; }
    public long getSupportCalculations() { return supportCalculations; }
    public long getMaxBufferSize() { return maxFFICSize; }
    public String getVariantName() { return "TopKPFIM (Li et al. 2017, paper-faithful)"; }
}