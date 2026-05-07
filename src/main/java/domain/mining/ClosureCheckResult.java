package domain.mining;

import domain.model.FrequentItemset;
import java.util.List;
import java.util.ArrayList;

public class ClosureCheckResult {
    
    private final boolean closed;
    private final List<FrequentItemset> extensions;

    public ClosureCheckResult(boolean closed, List<FrequentItemset> extensions) {
        this.closed = closed;
        this.extensions = extensions != null ? extensions : new ArrayList<>();
    }

    public boolean isClosed() {
        return closed;
    }

    public List<FrequentItemset> getExtensions() {
        return extensions;
    }
}
