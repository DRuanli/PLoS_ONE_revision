package experiment;

import infrastructure.persistence.UncertainDatabase;
import infrastructure.persistence.Vocabulary;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;

public class GenerateSensitivityData {

    private static final String[] BASE_DATASETS = {
            "processed_data/chess_uncertain.txt",
            "processed_data/liquor_11frequent_uncertain.txt",
            "processed_data/mushrooms_uncertain.txt",
            "processed_data/retail_uncertain.txt"
    };

    private static final double DEFAULT_PMIN = 0.1;
    private static final double DEFAULT_PMAX = 1.0;
    private static final double DEFAULT_ALPHA = 1.0;
    private static final double DEFAULT_RHO = 0.02;

    private static final double[] ALPHA_VALUES = { 0.5, 1.0, 1.5, 2.0 };
    private static final double[] RHO_VALUES   = { 0.0, 0.02, 0.05, 0.10 };
    private static final double[] PMIN_VALUES  = { 0.05, 0.1, 0.2, 0.3 };
    private static final double[] PMAX_VALUES  = { 0.7, 0.8, 0.9, 1.0 };

    public static void main(String[] args) throws Exception {
        ExperimentRunner.ensureDir("processed_data");

        for (String baseDataset : BASE_DATASETS) {
            String baseName = extractBaseName(baseDataset);
            System.out.println("Processing " + baseName);

            UncertainDatabase baseDb = UncertainDatabase.loadFromFile(baseDataset);

            for (double alpha : ALPHA_VALUES) {
                String outFile = String.format(Locale.US, "processed_data/%s_alpha_%.1f.txt", baseName, alpha);
                saveDatabase(baseDb, outFile);
                System.out.println("  Created: " + outFile);
            }

            for (double rho : RHO_VALUES) {
                String outFile = String.format(Locale.US, "processed_data/%s_rho_%.2f.txt", baseName, rho);
                saveDatabase(baseDb, outFile);
                System.out.println("  Created: " + outFile);
            }

            for (double pmin : PMIN_VALUES) {
                String outFile = String.format(Locale.US, "processed_data/%s_pmin_%.2f.txt", baseName, pmin);
                saveDatabase(baseDb, outFile);
                System.out.println("  Created: " + outFile);
            }

            for (double pmax : PMAX_VALUES) {
                String outFile = String.format(Locale.US, "processed_data/%s_pmax_%.1f.txt", baseName, pmax);
                saveDatabase(baseDb, outFile);
                System.out.println("  Created: " + outFile);
            }
        }

        System.out.println("\nDone. All sensitivity datasets generated.");
    }

    private static String extractBaseName(String path) {
        String name = path;
        int slash = Math.max(name.lastIndexOf('/'), name.lastIndexOf('\\'));
        if (slash >= 0) name = name.substring(slash + 1);
        int dot = name.lastIndexOf('.');
        if (dot > 0) name = name.substring(0, dot);
        return name;
    }

    private static void saveDatabase(UncertainDatabase db, String filename) throws IOException {
        Map<Integer, Map<String, Double>> transactionMap = new HashMap<>();
        Vocabulary vocab = db.getVocabulary();

        for (int item = 0; item < vocab.size(); item++) {
            String itemName = vocab.getItem(item);
            var tidset = db.getTidset(item);
            if (tidset.isEmpty()) continue;

            for (var entry : tidset.getEntries()) {
                int tid = entry.tid;
                double prob = entry.prob;
                transactionMap.computeIfAbsent(tid, k -> new HashMap<>()).put(itemName, prob);
            }
        }

        try (BufferedWriter w = new BufferedWriter(new FileWriter(filename))) {
            List<Integer> sortedTids = new ArrayList<>(transactionMap.keySet());
            sortedTids.sort(Integer::compareTo);

            for (int tid : sortedTids) {
                w.write(String.valueOf(tid));
                Map<String, Double> items = transactionMap.get(tid);
                for (Map.Entry<String, Double> e : items.entrySet()) {
                    w.write(" ");
                    w.write(e.getKey());
                    w.write(":");
                    w.write(String.format("%.4f", e.getValue()));
                }
                w.newLine();
            }
        }
    }
}
