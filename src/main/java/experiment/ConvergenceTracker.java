package experiment;

import java.util.*;

/**
 * ConvergenceTracker - Tracks threshold convergence speed during mining
 *
 * Measures how quickly the Top-K heap fills and how fast the minimum support
 * threshold converges to its final value. This demonstrates the advantage of
 * Best-First Search (BFS) over Depth-First Search (DFS).
 *
 * Key Metrics:
 * - Time and candidates to fill Top-K heap
 * - Time and candidates to reach 90%, 95%, 99% of final threshold
 * - Threshold evolution over time
 *
 * @author Dang Nguyen Le, Gia Huy Vo
 */
public class ConvergenceTracker {

    // Heap fill metrics
    private long candidatesToFillHeap = -1;
    private long timeToFillHeap = -1;  // Nanoseconds
    private boolean heapFilled = false;

    // Final threshold (set at end of mining)
    private int finalThreshold = 0;

    // Convergence milestones: percent -> candidate count
    private Map<Double, Long> convergenceMilestones = new HashMap<>();

    // Convergence times: percent -> nanoseconds
    private Map<Double, Long> convergenceTimes = new HashMap<>();

    // Threshold evolution snapshots
    private List<ThresholdSnapshot> snapshots = new ArrayList<>();

    // Track which convergence percentages we've already recorded
    private Set<Double> recordedPercentages = new HashSet<>();

    /**
     * Record when Top-K heap becomes full for the first time
     */
    public void recordHeapFilled(long candidatesProcessed, long timeNanos) {
        if (!heapFilled) {
            this.candidatesToFillHeap = candidatesProcessed;
            this.timeToFillHeap = timeNanos;
            this.heapFilled = true;
        }
    }

    /**
     * Record threshold snapshot during mining
     */
    public void recordThresholdSnapshot(int threshold, long candidatesProcessed, long timeNanos) {
        snapshots.add(new ThresholdSnapshot(candidatesProcessed, timeNanos, threshold, 0.0));

        // Check convergence milestones if finalThreshold is set
        if (finalThreshold > 0) {
            double convergencePercent = (threshold * 100.0) / finalThreshold;

            // Record convergence milestones
            checkAndRecordMilestone(90.0, convergencePercent, candidatesProcessed, timeNanos);
            checkAndRecordMilestone(95.0, convergencePercent, candidatesProcessed, timeNanos);
            checkAndRecordMilestone(99.0, convergencePercent, candidatesProcessed, timeNanos);
        }
    }

    /**
     * Set final threshold value (called after mining completes)
     * This also updates convergence percentages in snapshots
     */
    public void setFinalThreshold(int threshold) {
        this.finalThreshold = threshold;

        // Update convergence percentages in snapshots
        for (ThresholdSnapshot snapshot : snapshots) {
            snapshot.convergencePercent = getConvergencePercent(snapshot.threshold);
        }

        // Recalculate milestones now that we know final threshold
        recordedPercentages.clear();
        convergenceMilestones.clear();
        convergenceTimes.clear();

        for (ThresholdSnapshot snapshot : snapshots) {
            double percent = snapshot.convergencePercent;
            checkAndRecordMilestone(90.0, percent, snapshot.candidatesProcessed, snapshot.timeNanos);
            checkAndRecordMilestone(95.0, percent, snapshot.candidatesProcessed, snapshot.timeNanos);
            checkAndRecordMilestone(99.0, percent, snapshot.candidatesProcessed, snapshot.timeNanos);
        }
    }

    /**
     * Helper to check and record convergence milestone
     */
    private void checkAndRecordMilestone(double targetPercent, double currentPercent,
                                         long candidates, long time) {
        if (currentPercent >= targetPercent && !recordedPercentages.contains(targetPercent)) {
            convergenceMilestones.put(targetPercent, candidates);
            convergenceTimes.put(targetPercent, time);
            recordedPercentages.add(targetPercent);
        }
    }

    /**
     * Get convergence percentage for a given threshold value
     */
    public double getConvergencePercent(int currentThreshold) {
        if (finalThreshold == 0) return 0.0;
        return (currentThreshold * 100.0) / finalThreshold;
    }

    // Getters

    public long getCandidatesToFillHeap() {
        return candidatesToFillHeap;
    }

    public long getTimeToFillHeap() {
        return timeToFillHeap;
    }

    public long getCandidatesTo90Pct() {
        return convergenceMilestones.getOrDefault(90.0, -1L);
    }

    public long getCandidatesTo95Pct() {
        return convergenceMilestones.getOrDefault(95.0, -1L);
    }

    public long getCandidatesTo99Pct() {
        return convergenceMilestones.getOrDefault(99.0, -1L);
    }

    public long getTimeTo90Pct() {
        return convergenceTimes.getOrDefault(90.0, -1L);
    }

    public long getTimeTo95Pct() {
        return convergenceTimes.getOrDefault(95.0, -1L);
    }

    public long getTimeTo99Pct() {
        return convergenceTimes.getOrDefault(99.0, -1L);
    }

    public int getFinalThreshold() {
        return finalThreshold;
    }

    public List<ThresholdSnapshot> getSnapshots() {
        return new ArrayList<>(snapshots);
    }

    public boolean hasFilledHeap() {
        return heapFilled;
    }

    /**
     * Snapshot of threshold state at a point in time
     */
    public static class ThresholdSnapshot {
        public final long candidatesProcessed;
        public final long timeNanos;
        public final int threshold;
        public double convergencePercent;

        public ThresholdSnapshot(long candidates, long time, int threshold, double percent) {
            this.candidatesProcessed = candidates;
            this.timeNanos = time;
            this.threshold = threshold;
            this.convergencePercent = percent;
        }
    }
}