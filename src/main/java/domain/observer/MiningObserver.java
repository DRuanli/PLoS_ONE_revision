package domain.observer;

public interface MiningObserver {
    void onPhaseStart(int phase, String description);
    void onPhaseComplete(int phase, long durationMs);
}
