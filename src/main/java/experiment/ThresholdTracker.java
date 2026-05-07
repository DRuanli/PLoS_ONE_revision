package experiment;

import java.util.ArrayList;
import java.util.List;

public class ThresholdTracker {

    private final List<ThresholdSnapshot> snapshots;
    private long totalCandidates;
    private long candidatesProcessed;
    private int earlyTerminationPoint;
    private boolean terminated;
    private int finalThreshold;

    public ThresholdTracker() {
        this.snapshots = new ArrayList<>();
        this.totalCandidates = 0;
        this.candidatesProcessed = 0;
        this.earlyTerminationPoint = -1;
        this.terminated = false;
        this.finalThreshold = 0;
    }

    public void setTotalCandidates(long total) {
        this.totalCandidates = total;
    }

    public void recordSnapshot(int threshold) {
        candidatesProcessed++;
        double progress = totalCandidates > 0 ?
            (candidatesProcessed * 100.0 / totalCandidates) : 0;
        snapshots.add(new ThresholdSnapshot(candidatesProcessed, threshold, progress));
        this.finalThreshold = threshold;
    }

    public void recordEarlyTermination() {
        if (!terminated) {
            earlyTerminationPoint = (int) candidatesProcessed;
            terminated = true;
        }
    }

    public List<ThresholdSnapshot> getSnapshots() {
        return new ArrayList<>(snapshots);
    }

    public List<ThresholdSnapshot> getSnapshotsAtInterval(double intervalPercent) {
        List<ThresholdSnapshot> result = new ArrayList<>();
        double lastPercent = -intervalPercent;

        for (ThresholdSnapshot s : snapshots) {
            if (s.progressPercent >= lastPercent + intervalPercent) {
                result.add(s);
                lastPercent = s.progressPercent;
            }
        }

        if (!snapshots.isEmpty()) {
            ThresholdSnapshot last = snapshots.get(snapshots.size() - 1);
            if (result.isEmpty() || result.get(result.size() - 1) != last) {
                result.add(last);
            }
        }

        return result;
    }

    public long getCandidatesAt(double thresholdPercent) {
        if (snapshots.isEmpty() || finalThreshold == 0) return -1;

        int targetThreshold = (int) (finalThreshold * thresholdPercent / 100.0);

        for (ThresholdSnapshot s : snapshots) {
            if (s.threshold >= targetThreshold) {
                return s.candidatesProcessed;
            }
        }

        return candidatesProcessed;
    }

    public double getProgressAt(double thresholdPercent) {
        long candidates = getCandidatesAt(thresholdPercent);
        if (candidates < 0 || totalCandidates == 0) return -1;
        return (candidates * 100.0 / totalCandidates);
    }

    public int getEarlyTerminationPoint() {
        return earlyTerminationPoint;
    }

    public double getEarlyTerminationSavings() {
        if (earlyTerminationPoint < 0 || totalCandidates == 0) return 0;
        return ((totalCandidates - earlyTerminationPoint) * 100.0 / totalCandidates);
    }

    public long getTotalCandidates() {
        return totalCandidates;
    }

    public long getCandidatesProcessed() {
        return candidatesProcessed;
    }

    public int getFinalThreshold() {
        return finalThreshold;
    }

    public void reset() {
        snapshots.clear();
        totalCandidates = 0;
        candidatesProcessed = 0;
        earlyTerminationPoint = -1;
        terminated = false;
        finalThreshold = 0;
    }

    public static class ThresholdSnapshot {
        public final long candidatesProcessed;
        public final int threshold;
        public final double progressPercent;

        public ThresholdSnapshot(long candidates, int threshold, double progress) {
            this.candidatesProcessed = candidates;
            this.threshold = threshold;
            this.progressPercent = progress;
        }

        @Override
        public String toString() {
            return String.format("%.1f%%: threshold=%d (candidates=%d)",
                progressPercent, threshold, candidatesProcessed);
        }
    }
}