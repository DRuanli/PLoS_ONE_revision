package domain.observer;

public class PhaseTimingObserver implements MiningObserver{
    private long phase1Time = 0;
    private long phase2Time = 0;
    private long phase3Time = 0;

    @Override
    public void onPhaseStart(int phase, String description) {
        // Phase starting - could print status here if desired
    }

    @Override
    public void onPhaseComplete(int phase, long durationMs) {
        // Record the duration for this phase
        switch (phase) {
            case 1:
                phase1Time = durationMs;
                break;
            case 2:
                phase2Time = durationMs;
                break;
            case 3:
                phase3Time = durationMs;
                break;
        }
    }

    public long getPhase1Time() {
        return phase1Time;
    }

    public long getPhase2Time() {
        return phase2Time;
    }

    public long getPhase3Time() {
        return phase3Time;
    }

    public long getTotalTime() {
        return phase1Time + phase2Time + phase3Time;
    }
}
