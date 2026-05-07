package domain.mining;

public class PruningConfig {
    
    public boolean P1_earlyTermination = true;
    public boolean P2_thresholdPruning = true;
    public boolean P3_itemSupportThreshold = true;
    public boolean P4_subsetUpperBound = true;
    public boolean P5_upperBoundFilter = true;
    public boolean P6_tidsetSizePruning = true;
    public boolean P7_tidsetClosureSkip = true;
    
    public PruningConfig() {
    }
    
    public PruningConfig(boolean p1, boolean p2, boolean p3, boolean p4, 
                        boolean p5, boolean p6, boolean p7) {
        this.P1_earlyTermination = p1;
        this.P2_thresholdPruning = p2;
        this.P3_itemSupportThreshold = p3;
        this.P4_subsetUpperBound = p4;
        this.P5_upperBoundFilter = p5;
        this.P6_tidsetSizePruning = p6;
        this.P7_tidsetClosureSkip = p7;
    }
    
    public static PruningConfig full() {
        return new PruningConfig(true, true, true, true, true, true, true);
    }
    
    public static PruningConfig minimal() {
        return new PruningConfig(false, true, false, false, false, false, false);
    }
    
    public static PruningConfig none() {
        return new PruningConfig(false, false, false, false, false, false, false);
    }
    
    public PruningConfig withP1(boolean enabled) {
        this.P1_earlyTermination = enabled;
        return this;
    }
    
    public PruningConfig withP2(boolean enabled) {
        this.P2_thresholdPruning = enabled;
        return this;
    }
    
    public PruningConfig withP3(boolean enabled) {
        this.P3_itemSupportThreshold = enabled;
        return this;
    }
    
    public PruningConfig withP4(boolean enabled) {
        this.P4_subsetUpperBound = enabled;
        return this;
    }
    
    public PruningConfig withP5(boolean enabled) {
        this.P5_upperBoundFilter = enabled;
        return this;
    }
    
    public PruningConfig withP6(boolean enabled) {
        this.P6_tidsetSizePruning = enabled;
        return this;
    }
    
    public PruningConfig withP7(boolean enabled) {
        this.P7_tidsetClosureSkip = enabled;
        return this;
    }
    
    public int countEnabled() {
        int count = 0;
        if (P1_earlyTermination) count++;
        if (P2_thresholdPruning) count++;
        if (P3_itemSupportThreshold) count++;
        if (P4_subsetUpperBound) count++;
        if (P5_upperBoundFilter) count++;
        if (P6_tidsetSizePruning) count++;
        if (P7_tidsetClosureSkip) count++;
        return count;
    }
    
    public String getDescription() {
        if (countEnabled() == 7) return "full";
        if (countEnabled() == 1 && P2_thresholdPruning) return "minimal";
        if (countEnabled() == 0) return "none";
        
        StringBuilder sb = new StringBuilder();
        if (P1_earlyTermination) sb.append("P1,");
        if (P2_thresholdPruning) sb.append("P2,");
        if (P3_itemSupportThreshold) sb.append("P3,");
        if (P4_subsetUpperBound) sb.append("P4,");
        if (P5_upperBoundFilter) sb.append("P5,");
        if (P6_tidsetSizePruning) sb.append("P6,");
        if (P7_tidsetClosureSkip) sb.append("P7,");
        
        if (sb.length() > 0) {
            sb.setLength(sb.length() - 1);
        }
        return sb.toString();
    }
    
    @Override
    public String toString() {
        return String.format("PruningConfig[%s] P1=%b P2=%b P3=%b P4=%b P5=%b P6=%b P7=%b",
            getDescription(),
            P1_earlyTermination, P2_thresholdPruning, P3_itemSupportThreshold,
            P4_subsetUpperBound, P5_upperBoundFilter, P6_tidsetSizePruning, P7_tidsetClosureSkip);
    }
}
