package domain.mining;

import domain.support.SupportCalculator;
import infrastructure.persistence.UncertainDatabase;
import infrastructure.persistence.Vocabulary;
import domain.model.*;
import infrastructure.topK.TopKHeap;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

/**
 * AbstractMiner - Template Method Pattern for mining algorithms.
 *
 * ═══════════════════════════════════════════════════════════════════════════════
 * CHANGES FOR MAJOR REVISION (PLoS ONE):
 *   Added `pruningConfig` field (default = PruningConfig.full()) and
 *   `setPruningConfig()` setter so Experiment 4 (ablation) can disable
 *   individual pruning strategies P3–P7 without subclassing.
 *   P1 and P2 are controlled in the subclasses (TUFCI_V1, etc.).
 * ═══════════════════════════════════════════════════════════════════════════════
 *
 * @author Dang Nguyen Le, Gia Huy Vo
 */
public abstract class AbstractMiner {

    // ════════════════════════════════════════════════════════════════════════════
    // IMMUTABLE CONFIGURATION
    // ════════════════════════════════════════════════════════════════════════════
    private final UncertainDatabase database;
    private final double tau;
    private final int k;
    private final SupportCalculator calculator;
    private final Vocabulary vocab;

    // ════════════════════════════════════════════════════════════════════════════
    // MINING STATE (protected – direct subclass access for simplicity)
    // ════════════════════════════════════════════════════════════════════════════
    protected TopKHeap topK;
    protected Map<Itemset, CachedFrequentItemset> cache;
    protected Itemset[] singletonCache;
    protected int frequentItemCount;
    protected int[] frequentItems;
    protected experiment.ClosureMetrics closureMetrics;

    // ════════════════════════════════════════════════════════════════════════════
    // PRUNING CONFIGURATION  ← NEW: required for ablation study (Exp 4)
    // Default = full (all P1–P7 enabled).
    // P1 and P2 are checked inside each subclass (initializeTopK / executePhase3).
    // P3–P7 are checked inside checkClosureAndGenerateExtensions (this class).
    // ════════════════════════════════════════════════════════════════════════════
    protected PruningConfig pruningConfig = PruningConfig.full();

    /**
     * Sets the pruning configuration for this miner.
     * Must be called BEFORE mine() to take effect.
     * Used by Experiment 4 (ablation study) to disable individual strategies.
     *
     * @param config PruningConfig specifying which strategies are active
     */
    public void setPruningConfig(PruningConfig config) {
        if (config == null) throw new IllegalArgumentException("PruningConfig cannot be null");
        this.pruningConfig = config;
    }

    /** Returns the current pruning configuration. */
    public PruningConfig getPruningConfig() {
        return pruningConfig;
    }

    // ════════════════════════════════════════════════════════════════════════════
    // CONSTRUCTORS
    // ════════════════════════════════════════════════════════════════════════════

    public AbstractMiner(UncertainDatabase database, double tau, int k,
                         SupportCalculator calculator) {
        validateParameters(database, tau, k);
        if (calculator == null) throw new IllegalArgumentException("SupportCalculator cannot be null");
        this.database   = database;
        this.tau        = tau;
        this.k          = k;
        this.calculator = calculator;
        this.vocab      = database.getVocabulary();
        this.cache      = new HashMap<>();
    }

    public AbstractMiner(UncertainDatabase database, double tau, int k) {
        validateParameters(database, tau, k);
        this.database   = database;
        this.tau        = tau;
        this.k          = k;
        this.calculator = null;
        this.vocab      = database.getVocabulary();
        this.cache      = new HashMap<>();
    }

    private void validateParameters(UncertainDatabase database, double tau, int k) {
        if (database == null) throw new IllegalArgumentException("Database cannot be null");
        if (database.size() == 0) throw new IllegalArgumentException("Database cannot be empty");
        if (tau <= 0 || tau > 1)
            throw new IllegalArgumentException(String.format("Tau must be in (0, 1], got: %.4f", tau));
        if (k < 1)
            throw new IllegalArgumentException(String.format("k must be at least 1, got: %d", k));
    }

    // ════════════════════════════════════════════════════════════════════════════
    // PROTECTED GETTERS (immutable fields)
    // ════════════════════════════════════════════════════════════════════════════
    protected final UncertainDatabase getDatabase()  { return database;   }
    protected final double getTau()                   { return tau;        }
    protected final int getK()                        { return k;          }
    protected final SupportCalculator getCalculator() { return calculator; }
    protected final Vocabulary getVocabulary()        { return vocab;      }

    // ════════════════════════════════════════════════════════════════════════════
    // TEMPLATE METHOD
    // ════════════════════════════════════════════════════════════════════════════
    public final List<FrequentItemset> mine() {
        long start1 = System.nanoTime();
        List<FrequentItemset> frequent1Itemsets = computeAllSingletonSupports();
        long phase1Time = (System.nanoTime() - start1) / 1_000_000;

        long start2 = System.nanoTime();
        initializeTopKWithClosedSingletons(frequent1Itemsets);
        long phase2Time = (System.nanoTime() - start2) / 1_000_000;

        long start3 = System.nanoTime();
        executePhase3(frequent1Itemsets);
        long phase3Time = (System.nanoTime() - start3) / 1_000_000;

        return getTopKResults();
    }

    // ════════════════════════════════════════════════════════════════════════════
    // SHARED CONCRETE METHOD – Phase 1
    // ════════════════════════════════════════════════════════════════════════════
    protected List<FrequentItemset> computeAllSingletonSupports() {
        int vocabSize = getVocabulary().size();
        FrequentItemset[] resultArray = new FrequentItemset[vocabSize];
        ConcurrentHashMap<Itemset, CachedFrequentItemset> concurrentCache =
                new ConcurrentHashMap<>(vocabSize);

        this.singletonCache = new Itemset[vocabSize];
        for (int i = 0; i < vocabSize; i++) {
            singletonCache[i] = createSingletonItemset(i);
        }

        final UncertainDatabase db   = getDatabase();
        final SupportCalculator calc = getCalculator();

        java.util.stream.IntStream.range(0, vocabSize).parallel().forEach(item -> {
            Itemset singleton = singletonCache[item];
            Tidset tidset = db.getTidset(singleton);
            if (tidset.isEmpty()) { resultArray[item] = null; return; }

            double[] supportResult = calc.computeProbabilisticSupportFromTidset(tidset, db.size());
            int support       = (int) supportResult[0];
            double probability = supportResult[1];

            FrequentItemset fi = new FrequentItemset(singleton, support, probability);
            resultArray[item] = fi;
            concurrentCache.put(singleton,
                    new CachedFrequentItemset(singleton, support, probability, tidset));
        });

        List<FrequentItemset> result = Arrays.stream(resultArray)
                .filter(Objects::nonNull)
                .sorted(FrequentItemset::compareBySupport)
                .collect(Collectors.toList());

        this.cache = concurrentCache;
        return result;
    }

    // ════════════════════════════════════════════════════════════════════════════
    // UTILITY METHODS
    // ════════════════════════════════════════════════════════════════════════════
    protected Itemset createSingletonItemset(int item) {
        Itemset itemset = new Itemset(getVocabulary());
        itemset.add(item);
        return itemset;
    }

    protected int getThreshold() { return topK.getMinSupport(); }

    protected int getItemSupport(int item) {
        if (item < 0 || item >= singletonCache.length) return 0;
        Itemset singleton = singletonCache[item];
        CachedFrequentItemset cached = cache.get(singleton);
        return (cached != null) ? cached.getSupport() : 0;
    }

    protected int getMaxItemIndex(Itemset itemset) {
        int[] items = itemset.getItemsArray();
        if (items.length == 0) return -1;
        return items[items.length - 1];
    }

    // ════════════════════════════════════════════════════════════════════════════
    // CLOSURE CHECKING – Phase 2 (singletons)
    // ════════════════════════════════════════════════════════════════════════════
    protected boolean checkClosure1Itemset(FrequentItemset oneItemFI, int supOneItem,
                                           List<FrequentItemset> frequent1Itemset, int minsup) {
        int itemA = oneItemFI.getItems().get(0);

        for (FrequentItemset otherFI : frequent1Itemset) {
            int itemB = otherFI.getItems().get(0);
            if (itemA == itemB) continue;
            if (otherFI.getSupport() < supOneItem) break;

            Itemset unionItemset = oneItemFI.union(otherFI);
            CachedFrequentItemset cached = cache.get(unionItemset);
            int supAB;
            double probAB;
            Tidset tidsetAB;

            if (cached != null) {
                supAB    = cached.getSupport();
                probAB   = cached.getProbability();
                tidsetAB = cached.getTidset();
            } else {
                tidsetAB = cache.get(oneItemFI).getTidset()
                               .intersect(cache.get(otherFI).getTidset());
                if (!tidsetAB.isEmpty()) {
                    double[] result = getCalculator()
                            .computeProbabilisticSupportFromTidset(tidsetAB, getDatabase().size());
                    supAB  = (int) result[0];
                    probAB = result[1];
                } else {
                    supAB  = 0;
                    probAB = 0.0;
                }
                if (otherFI.getSupport() >= minsup) {
                    cache.put(unionItemset,
                            new CachedFrequentItemset(unionItemset, supAB, probAB, tidsetAB));
                }
            }

            if (supAB == supOneItem) return false;
        }
        return true;
    }

    // ════════════════════════════════════════════════════════════════════════════
    // CLOSURE CHECKING – Phase 3  ← KEY METHOD WITH PRUNING CONFIG INTEGRATION
    //
    // P3 – P7 now respect pruningConfig flags set via setPruningConfig().
    // This enables Experiment 4 (ablation) to isolate each strategy's contribution.
    // ════════════════════════════════════════════════════════════════════════════
    protected ClosureCheckResult checkClosureAndGenerateExtensions(
            FrequentItemset candidate, int threshold) {

        int supX = candidate.getSupport();
        boolean isClosed = true;
        List<FrequentItemset> extensions = new ArrayList<>();
        int maxItemInX = getMaxItemIndex(candidate);

        int totalExtensions    = 0;
        int extensionsExamined = 0;
        int violationPosition  = -1;

        if (closureMetrics != null) {
            for (int idx = 0; idx < frequentItemCount; idx++) {
                int item = frequentItems[idx];
                if (!candidate.contains(item) && item > maxItemInX) totalExtensions++;
            }
        }

        boolean closureCheckingDone = false;

        for (int idx = 0; idx < frequentItemCount; idx++) {
            int item = frequentItems[idx];
            if (candidate.contains(item)) continue;

            int itemSupport = getItemSupport(item);

            // ── P3: Item Support Threshold (configurable) ─────────────────────
            if (pruningConfig.P3_itemSupportThreshold && itemSupport < threshold) {
                break;
            }

            if (!closureCheckingDone && itemSupport < supX) {
                closureCheckingDone = true;
            }

            boolean needClosureCheck  = !closureCheckingDone && isClosed;
            boolean needExtension     = (item > maxItemInX);

            int upperBound = Math.min(supX, itemSupport);

            // ── P4: Subset-Based Upper Bound Tightening (configurable) ─────────
            if (pruningConfig.P4_subsetUpperBound && topK.isFull() && needExtension) {
                for (int existingItem : candidate.getItemsArray()) {
                    Itemset twoItemset = Itemset.of(vocab,
                            Math.min(existingItem, item),
                            Math.max(existingItem, item));
                    CachedFrequentItemset cachedSubset = cache.get(twoItemset);
                    if (cachedSubset != null) {
                        upperBound = Math.min(upperBound, cachedSubset.getSupport());
                        if (upperBound < threshold) break;
                    }
                }
            }

            // ── P5: Upper Bound Filtering (configurable) ────────────────────────
            boolean canEnterTopK = !pruningConfig.P5_upperBoundFilter || (upperBound >= threshold);
            boolean shouldGenerateExtension = needExtension && canEnterTopK;

            if (!needClosureCheck && !shouldGenerateExtension) continue;
            if (needExtension) extensionsExamined++;

            Itemset itemItemset = singletonCache[item];
            Itemset Xe = candidate.union(itemItemset);
            int supXe;
            double probXe;
            Tidset tidsetXe;

            CachedFrequentItemset cached = cache.get(Xe);
            if (cached != null) {
                supXe   = cached.getSupport();
                probXe  = cached.getProbability();
                tidsetXe = cached.getTidset();
            } else {
                CachedFrequentItemset xInfo    = cache.get(candidate);
                CachedFrequentItemset itemInfo = cache.get(itemItemset);

                if (xInfo == null || itemInfo == null) {
                    Tidset tidsetX    = getDatabase().getTidset(candidate);
                    Tidset tidsetItem = getDatabase().getTidset(itemItemset);
                    tidsetXe = tidsetX.intersect(tidsetItem);
                } else {
                    tidsetXe = xInfo.getTidset().intersect(itemInfo.getTidset());
                }

                int tidsetSize = tidsetXe.size();

                // ── P6: Tidset Size Pruning (configurable) ──────────────────────
                if (pruningConfig.P6_tidsetSizePruning && tidsetSize < threshold && !needClosureCheck) {
                    continue;
                }

                // ── P7: Tidset-Based Early Closure Detection (configurable) ─────
                if (pruningConfig.P7_tidsetClosureSkip && needClosureCheck && tidsetSize < supX) {
                    if (!shouldGenerateExtension) continue;
                    needClosureCheck = false;
                }

                double[] result = getCalculator().computeProbabilisticSupportFromTidset(
                        tidsetXe, getDatabase().size());
                supXe  = (int) result[0];
                probXe = result[1];
                cache.put(Xe, new CachedFrequentItemset(Xe, supXe, probXe, tidsetXe));
            }

            if (needClosureCheck && supXe == supX) {
                isClosed = false;
                if (violationPosition < 0) violationPosition = extensionsExamined;
            }

            if (shouldGenerateExtension) {
                extensions.add(new FrequentItemset(Xe, supXe, probXe));
            }
        }

        if (closureMetrics != null) {
            closureMetrics.recordClosureCheck(totalExtensions, extensionsExamined,
                                               isClosed, violationPosition);
        }

        return new ClosureCheckResult(isClosed, extensions);
    }

    // ════════════════════════════════════════════════════════════════════════════
    // ABSTRACT METHODS (subclasses implement)
    // ════════════════════════════════════════════════════════════════════════════
    protected abstract void initializeTopKWithClosedSingletons(List<FrequentItemset> singletonPatterns);
    protected abstract void executePhase3(List<FrequentItemset> singletonPatterns);
    protected abstract List<FrequentItemset> getTopKResults();
}