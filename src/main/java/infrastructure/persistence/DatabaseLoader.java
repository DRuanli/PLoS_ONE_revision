package infrastructure.persistence;

import java.io.*;
import java.util.*;

/**
 * DatabaseLoader - Utility for loading datasets and assigning random probabilities.
 *
 * Used by experiments to load standard datasets (without probabilities) and
 * convert them to uncertain databases by assigning random probabilities.
 */
public class DatabaseLoader {

    /**
     * Load dataset from file and assign random probabilities to items.
     *
     * Format: tid item1 item2 item3 ...
     * The loader will add random probabilities in [minProb, maxProb] range.
     *
     * @param filename path to dataset file
     * @param minProb minimum probability (e.g., 0.5)
     * @param maxProb maximum probability (e.g., 0.9)
     * @return uncertain database with random probabilities
     * @throws IOException if file cannot be read
     */
    public static UncertainDatabase loadWithUncertainty(String filename,
                                                        double minProb,
                                                        double maxProb) throws IOException {
        Vocabulary vocab = new Vocabulary();
        UncertainDatabase db = new UncertainDatabase(vocab);
        Random random = new Random(42); // Fixed seed for reproducibility

        try (BufferedReader br = new BufferedReader(new FileReader(filename))) {
            String line;
            while ((line = br.readLine()) != null) {
                line = line.trim();
                if (line.isEmpty()) continue;

                String[] parts = line.split("\\s+");
                if (parts.length < 2) continue;

                // First part is transaction ID
                int tid = Integer.parseInt(parts[0]);
                Map<Integer, Double> transaction = new HashMap<>();

                // Remaining parts are items - assign random probabilities
                for (int i = 1; i < parts.length; i++) {
                    String itemName = parts[i];
                    int itemIndex = vocab.getOrCreateIndex(itemName);

                    // Assign random probability in [minProb, maxProb]
                    double prob = minProb + (maxProb - minProb) * random.nextDouble();
                    transaction.put(itemIndex, prob);
                }

                db.addTransaction(tid, transaction);
            }
        }

        db.buildVerticalDatabase();

        // Extract dataset name from filename
        String datasetName = extractDatasetName(filename);
        db.setName(datasetName);

        return db;
    }

    /**
     * Extract dataset name from file path.
     * Examples:
     *   "datasets/chess.dat" -> "chess"
     *   "data/mushroom_uncertain.txt" -> "mushroom"
     */
    private static String extractDatasetName(String filename) {
        String name = filename;

        // Remove directory path
        int lastSlash = Math.max(name.lastIndexOf('/'), name.lastIndexOf('\\'));
        if (lastSlash >= 0) {
            name = name.substring(lastSlash + 1);
        }

        // Remove extension
        int lastDot = name.lastIndexOf('.');
        if (lastDot > 0) {
            name = name.substring(0, lastDot);
        }

        // Remove "_uncertain" suffix if present
        name = name.replace("_uncertain", "");

        return name;
    }

    /**
     * Load dataset with default probability range [0.5, 0.9].
     */
    public static UncertainDatabase load(String filename) throws IOException {
        return loadWithUncertainty(filename, 0.5, 0.9);
    }
}
