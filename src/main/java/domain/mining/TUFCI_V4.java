package domain.mining;

import domain.support.SupportCalculator;
import infrastructure.persistence.UncertainDatabase;
import domain.model.*;
import domain.support.DirectConvolutionSupportCalculator;
import infrastructure.topK.TopKHeap;

import java.util.*;

/**
 * TUFCI_V4: Depth-First Search + Search-Strategy Pruning (P1, P2, P3)
 *
 * <p>DFS counterpart of {@link TUFCI_V3}.  Retains search-strategy pruning
 * (P1–P3) while removing computation-level shortcuts (P4–P7).  The class
 * structure, field layout, statistics interface, and closure-check logic mirror
 * V3 exactly; the only differences are those <em>structurally required</em> by DFS:</p>
 * <ul>
 *   <li>A {@link Deque} (stack, LIFO) replaces the {@link PriorityQueue}.</li>
 *   <li>P2 uses {@code continue} instead of {@code break} — a DFS stack
 *       holds candidates with mixed support levels, so one below-threshold
 *       entry does not imply all remaining entries are also below threshold.</li>
 *   <li>Extensions are sorted ascending by support and pushed in that order
 *       so that the highest-support child is explored first (mirrors V2).</li>
 *   <li>{@link #maxStackSize} is tracked as a DFS-specific memory metric.</li>
 * </ul>
 *
 * <h2>Algorithm Characteristics</h2>
 * <ul>
 *   <li><b>Search strategy:</b>  Depth-First (Stack – LIFO order)</li>
 *   <li><b>Active pruning:</b>   P1 (Phase-2 term.), P2 (skip), P3 (item-loop break) + frontier filter</li>
 *   <li><b>Closure order:</b>    Descending support (same as V3)</li>
 *   <li><b>Early termination:</b> No — DFS cannot globally terminate on a
 *       single below-threshold candidate</li>
 * </ul>
 *
 * <h2>Pruning Status</h2>
 * <pre>
 * P1  Phase 2 Early Termination      ENABLED    stops singleton loop when heap full & sup < minsup
 * P2  Main Loop Threshold Pruning    ENABLED    skips below-threshold candidates (continue)
 * P3  Item Support Threshold         ENABLED    breaks item loop when support(i) < threshold
 * P4  Subset-Based Upper Bound       DISABLED   no 2-itemset bound tightening
 * P5  Upper Bound Filtering          DISABLED   no P4-tightened bound (see frontier filter)
 * P6  Tidset Size Pruning            DISABLED   tidset gating via frontier filter only
 * P7  Tidset-Based Closure Skip      DISABLED   closure decided only by anti-monotonicity flag
 * </pre>
 *
 * <h2>Search-frontier upper-bound filter (not one of P1–P7)</h2>
 * <p>After P3 breaks on {@code support(i) < threshold}, one additional
 * tidset-size check gates each extension before the expensive convolution:</p>
 * <ul>
 *   <li><b>Joint pre-check</b> – {@code |tidset(X∪{i})| < threshold}
 *       skips convolution (intersection already computed).  Exploits
 *       {@code support(X∪{i}) ≤ |tidset(X∪{i})|}</li>
 * </ul>
 * <p>This is a <em>search-frontier</em> decision (the extension never
 * enters the stack) rather than a per-candidate <em>computation
 * shortcut</em>.  P4–P7 shortcuts inside the item loop remain disabled.</p>
 *
 * <h2>V3 vs V4 at a glance (BFS vs DFS with identical pruning)</h2>
 * <pre>
 * ┌────────────────────────────┬──────────────────────────┬──────────────────────────┐
 * │ Aspect                     │ V3  (BFS + P1–P3)        │ V4  (DFS + P1–P3)        │
 * ├────────────────────────────┼──────────────────────────┼──────────────────────────┤
 * │ Data Structure             │ PriorityQueue            │ Stack (Deque)            │
 * │ Candidate Selection        │ Highest support first    │ Most recent (LIFO)       │
 * │ P2 Action                  │ break (global term)      │ continue (skip one)      │
 * │ Threshold Rise Speed       │ Fast                     │ Slower                   │
 * │ Memory                     │ O(candidates)            │ O(depth × branch)        │
 * │ Closure-check method       │ identical                │ identical                │
 * └────────────────────────────┴──────────────────────────┴──────────────────────────┘
 * </pre>
 *
 * @author Dang Nguyen Le, Gia Huy Vo
 */
public class TUFCI_V4 extends AbstractMiner {

    // ==================== Instance Variables ====================

    /** Stack for depth-first traversal in Phase 3. */
    private Deque<FrequentItemset> stack;

    // ── experiment statistics (mirrors V3 / V2 field layout) ──
    private long candidatesExplored  = 0;
    private long candidatesPruned    = 0;
    private long supportCalculations = 0;
    private long closureChecks       = 0;
    private long maxStackSize        = 0;
    private experiment.ThresholdTracker   thresholdTracker;
    private experiment.ConvergenceTracker convergenceTracker;

    // ==================== Constructors ====================

    public TUFCI_V4(UncertainDatabase database, double tau, int k) {
        super(database, tau, k, new DirectConvolutionSupportCalculator(tau));
    }

    public TUFCI_V4(UncertainDatabase database, double tau, int k, SupportCalculator calculator) {
        super(database, tau, k, calculator);
    }

    // ==================== Tracker / Metrics Setters ====================

    public void setThresholdTracker(experiment.ThresholdTracker tracker) {
        this.thresholdTracker = tracker;
    }

    public void setClosureMetrics(experiment.ClosureMetrics metrics) {
        this.closureMetrics = metrics;   // inherited protected field
    }

    public void setConvergenceTracker(experiment.ConvergenceTracker tracker) {
        this.convergenceTracker = tracker;
    }

    // ==================== Phase 2: Initialize ====================

    /**
     * Initialises the Top-K heap, closure-checks singletons with P1 early
     * termination, and seeds the stack with 2-itemsets.
     *
     * <p><b>P1 is ENABLED.</b>  The loop stops once the heap is full and the
     * current singleton support falls below {@code minsup}, identical to V2.</p>
     *
     * @param frequent1Itemsets all singletons from Phase 1, sorted support DESC
     */
    @Override
    protected void initializeTopKWithClosedSingletons(List<FrequentItemset> frequent1Itemsets) {
        this.topK  = new TopKHeap(getK());
        this.stack = new ArrayDeque<>();

        int minsup = 0;
        int processedItemCount = 0;
        long phase2StartTime = System.nanoTime();

        // Process 1-itemsets with P1: Early Termination
        for (FrequentItemset fi : frequent1Itemsets) {
            int support = fi.getSupport();

            // P1: Phase 2 Early Termination
            if (topK.isFull() && support < minsup) {
                break;
            }

            processedItemCount++;
            closureChecks++;

            boolean isClosed = checkClosure1Itemset(fi, support, frequent1Itemsets, minsup);

            if (isClosed) {
                boolean wasNotFull = !topK.isFull();
                boolean inserted   = topK.insert(fi);
                if (inserted && topK.isFull()) {
                    minsup = topK.getMinSupport();
                    if (convergenceTracker != null && wasNotFull) {
                        long elapsed = System.nanoTime() - phase2StartTime;
                        convergenceTracker.recordHeapFilled(processedItemCount, elapsed);
                    }
                }
            }
        }

        // Build frequent-items array (support ≥ final minsup; order = DESC)
        List<Integer> frequentItemIndices = new ArrayList<>();
        for (int i = 0; i < processedItemCount; i++) {
            FrequentItemset fi = frequent1Itemsets.get(i);
            if (fi.getSupport() >= minsup) {
                frequentItemIndices.add(fi.getItems().get(0));
            }
        }

        this.frequentItemCount = frequentItemIndices.size();
        this.frequentItems     = new int[frequentItemCount];
        for (int i = 0; i < frequentItemCount; i++) {
            frequentItems[i] = frequentItemIndices.get(i);
        }

        // Seed stack with cached 2-itemsets that meet the final threshold.
        // Sorted ascending so that after push the highest-support 2-itemset
        // sits on top — mirrors V2's seeding strategy.
        List<FrequentItemset> twoItemsets = new ArrayList<>();
        for (Map.Entry<Itemset, CachedFrequentItemset> entry : cache.entrySet()) {
            Itemset itemset = entry.getKey();
            if (itemset.size() == 2) {
                CachedFrequentItemset cached = entry.getValue();
                if (cached.getSupport() >= minsup) {
                    twoItemsets.add(cached.toFrequentItemset());
                }
            }
        }

        twoItemsets.sort((a, b) -> Integer.compare(a.getSupport(), b.getSupport()));
        for (FrequentItemset fi : twoItemsets) {
            stack.push(fi);
        }

        maxStackSize = stack.size();
    }

    // ==================== Phase 3: DFS Mining ====================

    /**
     * Depth-first search loop with P2 skip pruning.
     *
     * <p>Unlike BFS, the stack does not guarantee support-descending order.  A
     * candidate below threshold may be followed by one above threshold, so P2
     * can only {@code continue} past the current entry — it cannot terminate the
     * entire search.  This is the key structural difference from {@link TUFCI_V3}.</p>
     *
     * <p>Inside each iteration {@link #checkClosureAndGenerateExtensionsSearchOnly}
     * is used — the same method body as V3, ensuring the closure-check logic
     * is identical across the BFS/DFS comparison.</p>
     *
     * @param frequent1itemsets unused; required by the template-method signature
     */
    @Override
    protected void executePhase3(List<FrequentItemset> frequent1itemsets) {
        candidatesExplored = 0;
        candidatesPruned   = 0;
        long phase3StartTime = System.nanoTime();

        while (!stack.isEmpty()) {
            // Track max stack size (DFS-specific memory metric)
            if (stack.size() > maxStackSize) {
                maxStackSize = stack.size();
            }

            FrequentItemset candidate = stack.pop();
            candidatesExplored++;

            int threshold = getThreshold();

            // ★ KEY [Search]: DFS — P2 continue = skip one only
            // ── P2: Main Loop Threshold Pruning (ENABLED — continue) ────────
            // DFS cannot early-terminate: later stack entries may have support
            // above the current threshold.  Skip this candidate only.
            if (candidate.getSupport() < threshold) {
                candidatesPruned++;
                continue;
            }
            // ─────────────────────────────────────────────────────────────

            closureChecks++;
            // ★ KEY [Pruning]: Search-only — P4/P5/P6/P7 disabled
            ClosureCheckResult result = checkClosureAndGenerateExtensionsSearchOnly(candidate);

            if (result.isClosed()) {
                topK.insert(candidate);
            }

            // Convergence snapshot every 100 candidates (matches V2 cadence)
            if (convergenceTracker != null && candidatesExplored % 100 == 0) {
                long elapsed = System.nanoTime() - phase3StartTime;
                convergenceTracker.recordThresholdSnapshot(getThreshold(), candidatesExplored, elapsed);
            }

            // Filter extensions by the (possibly updated) threshold, then
            // push in ascending-support order so highest-support child is
            // explored first — mirrors V2's extension-push strategy.
            int newThreshold = getThreshold();
            List<FrequentItemset> validExtensions = new ArrayList<>();

            for (FrequentItemset ext : result.getExtensions()) {
                if (ext.getSupport() >= newThreshold) {
                    validExtensions.add(ext);
                } else {
                    candidatesPruned++;
                }
            }

            validExtensions.sort((a, b) -> Integer.compare(a.getSupport(), b.getSupport()));
            for (FrequentItemset ext : validExtensions) {
                stack.push(ext);
            }
        }

        if (convergenceTracker != null) {
            convergenceTracker.setFinalThreshold(getThreshold());
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //  SEARCH-STRATEGY CLOSURE CHECK & EXTENSION GENERATION  (P3 only; no P4–P7)
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * Checks closure and generates canonical extensions with P3 but <em>without</em>
     * P4–P7.  Logic is identical to {@link TUFCI_V3#checkClosureAndGenerateExtensionsSearchOnly}.
     *
     * <p>Items are visited in support-descending order.  P3 breaks the loop once
     * {@code support(i) < threshold}.  For each surviving item {@code i}:</p>
     * <ol>
     *   <li>If the anti-monotonicity flag has not yet been set and {@code i} could
     *       still violate closure ({@code support({i}) ≥ supX}), support of
     *       {@code X ∪ {i}} is computed and compared to {@code supX}.</li>
     *   <li>If {@code i > max(X)} (canonical order) an extension is added —
     *       support is computed if not already in step 1 or in the cache.</li>
     * </ol>
     *
     * <p><b>Disabled shortcuts (vs {@link AbstractMiner#checkClosureAndGenerateExtensions}):</b></p>
     * <ul>
     *   <li><b>P4</b> – no upper-bound tightening via cached 2-itemset supports.</li>
     *   <li><b>P5</b> – no P4-tightened upper-bound filtering; extensions are
     *       gated only by the joint tidset-size frontier check.</li>
     *   <li><b>P6</b> – no singleton-tidset shortcut (subsumed by P3).</li>
     *   <li><b>P7</b> – no tidset-based closure skip; anti-monotonicity flag only.</li>
     * </ul>
     *
     * @param candidate pattern {@code X} whose closure is being tested
     * @return closure status and the list of all canonical extensions
     */
    private ClosureCheckResult checkClosureAndGenerateExtensionsSearchOnly(FrequentItemset candidate) {
        int supX         = candidate.getSupport();
        int threshold    = getThreshold();   // stays constant; no TopK insertion during this check
        boolean isClosed = true;
        List<FrequentItemset> extensions = new ArrayList<>();
        int maxItemInX   = getMaxItemIndex(candidate);

        // ── metrics counters ──
        int totalExtensions    = 0;
        int extensionsExamined = 0;
        int violationPosition  = -1;

        if (closureMetrics != null) {
            for (int idx = 0; idx < frequentItemCount; idx++) {
                int item = frequentItems[idx];
                if (!candidate.contains(item) && item > maxItemInX) {
                    totalExtensions++;
                }
            }
        }

        /**
         * Anti-monotonicity flag.
         *
         * frequentItems is sorted by support DESC.  Once we reach an item whose
         * singleton support is strictly less than supX, every subsequent item also
         * satisfies support({i}) < supX.  By anti-monotonicity:
         *   support(X ∪ {i}) ≤ support({i}) < supX
         * so no later item can produce a closure violation.  The flag lets us stop
         * the closure check early without computing support.
         *
         * This is a mathematical property of the sorted iteration order, NOT one
         * of the P3–P7 pruning strategies.  (P7 short-circuits on *tidset size*;
         * this flag short-circuits on *item singleton support*.)
         */
        boolean closureCheckingDone = false;

        // Iterate frequent items — P3 breaks when support(i) < threshold
        for (int idx = 0; idx < frequentItemCount; idx++) {
            int item = frequentItems[idx];

            if (candidate.contains(item)) continue;

            int itemSupport = getItemSupport(item);

            // P3: Item Support Threshold — items sorted DESC so all
            // remaining items also have support < threshold.
            // support(X∪{i}) ≤ support({i}) < threshold  ⟹  prune.
            if (itemSupport < threshold) {
                break;
            }

            // Update anti-monotonicity flag (mathematical, not P3 — see block comment)
            if (!closureCheckingDone && itemSupport < supX) {
                closureCheckingDone = true;
            }

            boolean needClosureCheck = !closureCheckingDone && isClosed;
            boolean needExtension    = (item > maxItemInX);

            // ── P4 DISABLED: no upper-bound computation ──────────────────
            // ── P5 DISABLED: shouldGenerateExtension = needExtension ──────
            // ─────────────────────────────────────────────────────────────

            // Nothing to do for this item — skip cheaply
            if (!needClosureCheck && !needExtension) {
                continue;
            }

            if (needExtension) {
                extensionsExamined++;
            }

            // ────────────────────────────────────────────────────────────────
            //  Compute support(X ∪ {item})
            // ────────────────────────────────────────────────────────────────
            Itemset itemItemset = singletonCache[item];

            // Lookup singleton cache entry (tidset needed for intersection below)
            CachedFrequentItemset itemInfo = cache.get(itemItemset);

            Itemset Xe = candidate.union(itemItemset);

            int    supXe  = 0;
            double probXe = 0.0;

            CachedFrequentItemset cached = cache.get(Xe);
            if (cached != null) {
                // Cache hit – reuse (populated during Phase 2 or an earlier iteration)
                supXe  = cached.getSupport();
                probXe = cached.getProbability();
            } else {
                // Cache miss – intersect tidsets
                CachedFrequentItemset xInfo = cache.get(candidate);
                // itemInfo already looked up above; reuse

                Tidset tidsetXe;
                if (xInfo == null || itemInfo == null) {
                    // Fallback: pull tidsets directly from the database
                    tidsetXe = getDatabase().getTidset(candidate)
                                           .intersect(getDatabase().getTidset(itemItemset));
                } else {
                    tidsetXe = xInfo.getTidset().intersect(itemInfo.getTidset());
                }

                // ── Joint tidset upper-bound check ─────────────────────────
                // |tidset(X∪{i})| < threshold  ⟹  support < threshold ≤ supX.
                // Extension excluded from frontier; cannot violate closure.
                // NOT cached — support was never computed; avoids storing a
                // misleading placeholder (cf. the cache-poisoning fix in
                // AbstractMiner P6/P7).
                if (tidsetXe.size() < threshold) {
                    continue;
                }

                // ── Tidset closure gate ───────────────────────────────────
                // |tidset(X∪{i})| is a tight upper bound on support(X∪{i}).
                // If tidsetXe.size() < supX then support(Xe) < supX is
                // guaranteed  ⟹  closure violation (supXe == supX) impossible.
                //   • !needExtension  →  nothing left to do; skip convolution.
                //   •  needExtension  →  convolution still needed for the
                //                        support value; closure check dropped.
                if (needClosureCheck && tidsetXe.size() < supX) {
                    if (!needExtension) {
                        continue;   // no violation possible, no extension needed
                    }
                    needClosureCheck = false;   // extension still needed; closure moot
                }
                supportCalculations++;
                double[] result = getCalculator().computeProbabilisticSupportFromTidset(
                        tidsetXe, getDatabase().size());
                supXe  = (int) result[0];
                probXe = result[1];

                // Cache for reuse in deeper search levels
                cache.put(Xe, new CachedFrequentItemset(Xe, supXe, probXe, tidsetXe));
            }

            // ── Cached entry now below threshold ──────────────────────────
            // Threshold may have risen since this entry was cached.
            if (supXe < threshold) {
                continue;
            }

            // ── Closure violation check ────────────────────────────────────
            if (needClosureCheck && supXe == supX) {
                isClosed = false;
                if (violationPosition < 0) {
                    violationPosition = extensionsExamined;
                }
            }

            // ── Canonical extension ────────────────────────────────────────
            // P5 DISABLED: extension is added unconditionally — no upper-bound gate
            if (needExtension) {
                extensions.add(new FrequentItemset(Xe, supXe, probXe));
            }
        }

        if (closureMetrics != null) {
            closureMetrics.recordClosureCheck(totalExtensions, extensionsExamined, isClosed, violationPosition);
        }

        return new ClosureCheckResult(isClosed, extensions);
    }

    // ==================== Results ====================

    @Override
    protected List<FrequentItemset> getTopKResults() {
        List<FrequentItemset> results = topK.getAll();
        results.sort(FrequentItemset::compareBySupport);
        return results;
    }

    // ==================== Statistics (mirrors V2 / V3 interface) ====================

    public long getCandidatesExplored()   { return candidatesExplored; }
    public long getCandidatesPruned()     { return candidatesPruned; }
    public long getSupportCalculations()  { return supportCalculations; }
    public long getClosureChecks()        { return closureChecks; }
    public long getMaxStackSize()         { return maxStackSize; }

    public String getVariantName()        { return "V4 (DFS + P1-P3)"; }
}
