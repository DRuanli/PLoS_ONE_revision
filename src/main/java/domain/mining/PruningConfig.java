package domain.mining;

public class PruningConfig {
    public boolean P1_earlyTermination = true;
    public boolean P2_thresholdPruning = true;
    public boolean P3_itemSupportThreshold = true;
    public boolean P4_subsetUpperBound = true;
    public boolean P5_upperBoundFilter = true;
    public boolean P6_tidsetSizePruning = true;
    public boolean P7_tidsetClosureSkip = true;

    public PruningConfig() {}

    public PruningConfig(boolean p1, boolean p2, boolean p3, boolean p4, boolean p5, boolean p6, boolean p7) {
        this.P1_earlyTermination = p1;
        this.P2_thresholdPruning = p2;
        this.P3_itemSupportThreshold = p3;
        this.P4_subsetUpperBound = p4;
        this.P5_upperBoundFilter = p5;
        this.P6_tidsetSizePruning = p6;
        this.P7_tidsetClosureSkip = p7;
    }

    public static PruningConfig full() { return new PruningConfig(true, true, true, true, true, true, true); }
    public static PruningConfig minimal() { return new PruningConfig(false, true, false, false, false, false, false); }
    public static PruningConfig none() { return new PruningConfig(false, false, false, false, false, false, false); }

    public PruningConfig withP1(boolean e) { this.P1_earlyTermination = e; return this; }
    public PruningConfig withP2(boolean e) { this.P2_thresholdPruning = e; return this; }
    public PruningConfig withP3(boolean e) { this.P3_itemSupportThreshold = e; return this; }
    public PruningConfig withP4(boolean e) { this.P4_subsetUpperBound = e; return this; }
    public PruningConfig withP5(boolean e) { this.P5_upperBoundFilter = e; return this; }
    public PruningConfig withP6(boolean e) { this.P6_tidsetSizePruning = e; return this; }
    public PruningConfig withP7(boolean e) { this.P7_tidsetClosureSkip = e; return this; }

    @Override public String toString() { return "PruningConfig"; }
}