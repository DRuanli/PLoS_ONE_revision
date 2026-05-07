package presentation;

import domain.mining.*;
import domain.model.FrequentItemset;
import infrastructure.persistence.UncertainDatabase;

import java.io.IOException;
import java.util.List;

/**
 * Test runner for all TUFCI variants.
 * Usage: java -cp bin presentation.TestVariants <variant> <datafile> <tau> <k>
 *
 * Variants: V1, V1a, V1b, V2, V3, V4, BFS, DFS
 */
public class TestVariants {

    public static void main(String[] args) throws IOException {
        if (args.length < 4) {
            System.out.println("Usage: java -cp bin presentation.TestVariants <variant> <datafile> <tau> <k>");
            System.out.println();
            System.out.println("Variants:");
            System.out.println("  V1   - Best-First + Full Pruning + Descending");
            System.out.println("  V1a  - Best-First + Full Pruning + Random");
            System.out.println("  V1b  - Best-First + Full Pruning + Ascending");
            System.out.println("  V2   - DFS + Full Pruning");
            System.out.println("  V3   - Best-First + Minimal Pruning");
            System.out.println("  V4   - DFS + Minimal Pruning");
            System.out.println("  V5   - Best-First + Computation Pruning (P4-P7)");
            System.out.println("  V6   - DFS + Computation Pruning (P4-P7)");
            System.out.println("  BFS  - TUFCI_BFS (if exists)");
            System.out.println("  DFS  - TUFCI_DFS (if exists)");
            System.out.println();
            System.out.println("Example:");
            System.out.println("  java -cp bin presentation.TestVariants V1a data/kosarak_uncertain.txt 0.7 10");
            return;
        }

        String variant = args[0];
        String datafile = args[1];
        double tau = Double.parseDouble(args[2]);
        int k = Integer.parseInt(args[3]);

        System.out.println("╔═══════════════════════════════════════════════════════════╗");
        System.out.println("║        TUFCI Variant Comparison - " + variant + "                    ║");
        System.out.println("╚═══════════════════════════════════════════════════════════╝");
        System.out.println();
        System.out.println("Configuration:");
        System.out.println("  Variant           : " + variant);
        System.out.println("  Database file     : " + datafile);
        System.out.println("  Tau (threshold)   : " + tau);
        System.out.println("  K (top patterns)  : " + k);
        System.out.println();

        // Load database
        System.out.println("Loading database...");
        UncertainDatabase database = UncertainDatabase.loadFromFile(datafile);
        System.out.println("  Transactions : " + database.size());
        System.out.println("  Vocabulary   : " + database.getVocabulary().size() + " unique items");
        System.out.println();

        if (variant.equals("UTKFIM")) {
            UTKFIM miner = new UTKFIM(database, tau, k);
            System.out.println("Starting mining process...");
            System.out.println("─".repeat(65));

            long startTime = System.nanoTime();
            List<FrequentItemset> results = miner.mine();
            long endTime = System.nanoTime();
            long executionTime = (endTime - startTime) / 1_000_000; // Convert to ms

            System.out.println("─".repeat(65));
            System.out.println("Mining completed!");
            System.out.println();

            // Display results
            System.out.println("Performance Metrics:");
            System.out.println("  Execution Time : " + executionTime + " ms");
            System.out.println("  Patterns Found : " + results.size());
            System.out.println();

            System.out.println("Top-" + Math.min(k, results.size()) + " Patterns:");
            for (int i = 0; i < Math.min(k, results.size()); i++) {
                FrequentItemset pattern = results.get(i);
                System.out.printf("  %2d. %s\n", (i + 1), pattern);
            }
        }
        else {
            // Create miner based on variant
            AbstractMiner miner = createMiner(variant, database, tau, k);

            System.out.println("Starting mining process...");
            System.out.println("─".repeat(65));

            long startTime = System.nanoTime();
            List<FrequentItemset> results = miner.mine();
            long endTime = System.nanoTime();
            long executionTime = (endTime - startTime) / 1_000_000; // Convert to ms

            System.out.println("─".repeat(65));
            System.out.println("Mining completed!");
            System.out.println();

            // Display results
            System.out.println("Performance Metrics:");
            System.out.println("  Execution Time : " + executionTime + " ms");
            System.out.println("  Patterns Found : " + results.size());
            System.out.println();

            System.out.println("Top-" + Math.min(k, results.size()) + " Patterns:");
            for (int i = 0; i < Math.min(k, results.size()); i++) {
                FrequentItemset pattern = results.get(i);
                System.out.printf("  %2d. %s\n", (i + 1), pattern);
            }
        }
    }

    private static AbstractMiner createMiner(String variant, UncertainDatabase db, double tau, int k) {
        switch (variant.toUpperCase()) {
            case "V1":
                return new TUFCI_V1(db, tau, k);
            case "V2":
                return new TUFCI_V2(db, tau, k);
            case "V3":
                return new TUFCI_V3(db, tau, k);
            case "V4":
                return new TUFCI_V4(db, tau, k);
                /**
            case "V5":
                return new TUFCI_V5(db, tau, k);
            case "V6":
                return new TUFCI_V6(db, tau, k);
                 */
            default:
                System.err.println("Unknown variant: " + variant);
                System.err.println("Using default TUFCI");
                return new TUFCI(db, tau, k);
        }
    }
}
