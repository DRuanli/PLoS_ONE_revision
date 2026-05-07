package experiment;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class ClosureMetrics {

    private final List<ClosureCheckRecord> records;
    private long totalChecks;
    private long totalExtensionsExamined;
    private long violationsFound;
    private long closedPatterns;

    public ClosureMetrics() {
        this.records = new ArrayList<>();
        this.totalChecks = 0;
        this.totalExtensionsExamined = 0;
        this.violationsFound = 0;
        this.closedPatterns = 0;
    }

    public void recordClosureCheck(int totalExtensions, int extensionsExamined,
                                   boolean isClosed, int violationPosition) {
        records.add(new ClosureCheckRecord(
            totalExtensions, extensionsExamined, isClosed, violationPosition));

        totalChecks++;
        totalExtensionsExamined += extensionsExamined;

        if (isClosed) {
            closedPatterns++;
        } else {
            violationsFound++;
        }
    }

    public double getAverageExtensionsExamined() {
        return totalChecks > 0 ? (double) totalExtensionsExamined / totalChecks : 0;
    }

    public double getAverageExtensionsForViolations() {
        long violationExtensions = 0;
        long count = 0;

        for (ClosureCheckRecord r : records) {
            if (!r.isClosed) {
                violationExtensions += r.extensionsExamined;
                count++;
            }
        }

        return count > 0 ? (double) violationExtensions / count : 0;
    }

    public double getAverageExtensionsForClosed() {
        long closedExtensions = 0;
        long count = 0;

        for (ClosureCheckRecord r : records) {
            if (r.isClosed) {
                closedExtensions += r.extensionsExamined;
                count++;
            }
        }

        return count > 0 ? (double) closedExtensions / count : 0;
    }

    public Map<Integer, Integer> getViolationPositionDistribution() {
        Map<Integer, Integer> distribution = new HashMap<>();

        for (ClosureCheckRecord r : records) {
            if (!r.isClosed) {
                distribution.merge(r.violationPosition, 1, Integer::sum);
            }
        }

        return distribution;
    }

    public double getPercentViolationsInFirstN(int n) {
        if (violationsFound == 0) return 0;

        long count = 0;
        for (ClosureCheckRecord r : records) {
            if (!r.isClosed && r.violationPosition <= n) {
                count++;
            }
        }

        return (count * 100.0 / violationsFound);
    }

    public double getWorkSavedPercent() {
        long totalPossibleWork = 0;
        long actualWork = 0;

        for (ClosureCheckRecord r : records) {
            totalPossibleWork += r.totalExtensions;
            actualWork += r.extensionsExamined;
        }

        if (totalPossibleWork == 0) return 0;
        return ((totalPossibleWork - actualWork) * 100.0 / totalPossibleWork);
    }

    public long getTotalChecks() {
        return totalChecks;
    }

    public long getViolationsFound() {
        return violationsFound;
    }

    public long getClosedPatterns() {
        return closedPatterns;
    }

    public List<ClosureCheckRecord> getRecords() {
        return new ArrayList<>(records);
    }

    public void reset() {
        records.clear();
        totalChecks = 0;
        totalExtensionsExamined = 0;
        violationsFound = 0;
        closedPatterns = 0;
    }

    public static class ClosureCheckRecord {
        public final int totalExtensions;
        public final int extensionsExamined;
        public final boolean isClosed;
        public final int violationPosition;

        public ClosureCheckRecord(int total, int examined, boolean closed, int position) {
            this.totalExtensions = total;
            this.extensionsExamined = examined;
            this.isClosed = closed;
            this.violationPosition = position;
        }
    }

    public enum ClosureOrder {
        DESCENDING,
        RANDOM,
        ASCENDING
    }
}