package presentation;

import domain.model.FrequentItemset;
import domain.observer.PhaseTimingObserver;

import java.util.List;

public class Usage {
    public static void printUsage() {
        System.out.println("╔═══════════════════════════════════════════════════════════╗");
        System.out.println("║          TUFCI: Top-K Uncertain Frequent Closed           ║");
        System.out.println("║              Itemset Mining Algorithm                     ║");
        System.out.println("╚═══════════════════════════════════════════════════════════╝");
        System.out.println();
        System.out.println("Usage:");
        System.out.println("  java Test <database_file> [tau] [k]");
        System.out.println();
        System.out.println("Arguments:");
        System.out.println("  database_file : Path to the uncertain database file (required)");
        System.out.println("  tau          : Probability threshold (optional, default: 0.7)");
        System.out.println("  k            : Number of top patterns to mine (optional, default: 5)");
        System.out.println();
        System.out.println("Examples:");
        System.out.println("  java Test data/database.txt");
        System.out.println("  java Test data/database.txt 0.7");
        System.out.println("  java Test data/database.txt 0.7 10");
        System.out.println();
    }

    public static void printResults(List<FrequentItemset> results, int k) {
        // Check if any patterns were found
        if (results.isEmpty()) {
            System.out.println("╔═══════════════════════════════════════════════════════════╗");
            System.out.println("║                    No Patterns Found                      ║");
            System.out.println("╚═══════════════════════════════════════════════════════════╝");
            System.out.println();
            System.out.println("Suggestions:");
            System.out.println("  • Try lowering the tau threshold");
            System.out.println("  • Try increasing k value");
            System.out.println("  • Check if your database has enough transactions");
            return;
        }

        // Determine how many patterns to display (max 50 for readability)
        int displayCount = Math.min(results.size(), 50);

        // Print header
        System.out.println("╔═══════════════════════════════════════════════════════════╗");
        System.out.println("║                     Mining Results                        ║");
        System.out.println("╚═══════════════════════════════════════════════════════════╝");
        System.out.println();
        System.out.println("Top " + displayCount + " Patterns:");
        System.out.println();

        // Print table header
        System.out.printf("%-4s  %-40s  %8s  %12s%n",
            "Rank", "Itemset", "Support", "Probability");
        System.out.println("─".repeat(70));

        // Print each pattern
        for (int i = 0; i < displayCount; i++) {
            FrequentItemset fi = results.get(i);

            /**
             * For each pattern, we display:
             * - Rank: Position in the top-k (1-based)
             * - Itemset: The set of items (e.g., {milk, bread})
             * - Support: Expected number of transactions containing this itemset
             * - Probability: Likelihood of appearing in at least one transaction
             */
            System.out.printf("%-4d  %-40s  %8d  %12.4f%n",
                i + 1,                              // Rank (1-based index)
                fi.toStringWithCodec(),             // Itemset as readable string (inherited method)
                fi.getSupport(),                    // Expected support
                fi.getProbability()                 // Probability
            );
        }

        // Show indicator if there are more patterns
        if (results.size() > displayCount) {
            System.out.println("─".repeat(70));
            System.out.println("... and " + (results.size() - displayCount) + " more patterns");
        }

        System.out.println();

        // Print summary statistics
        System.out.println("Summary:");
        System.out.println("  Total patterns found   : " + results.size());
        System.out.println("  Patterns displayed     : " + displayCount);

        if (!results.isEmpty()) {
            FrequentItemset topPattern = results.get(0);
            System.out.println("  Highest support        : " + topPattern.getSupport());
            System.out.println("  Top pattern            : " + topPattern.toStringWithCodec());
        }

        System.out.println();
    }

    public static void printPerformanceMetrics(PhaseTimingObserver observer,
                                                long totalTime,
                                                long memoryUsed,
                                                int transactionCount,
                                                int patternCount) {
        System.out.println("╔═══════════════════════════════════════════════════════════╗");
        System.out.println("║                  Performance Metrics                      ║");
        System.out.println("╚═══════════════════════════════════════════════════════════╝");
        System.out.println();

        // Phase timing breakdown
        long phase1Time = observer.getPhase1Time();
        long phase2Time = observer.getPhase2Time();
        long phase3Time = observer.getPhase3Time();

        System.out.println("Execution Time Breakdown:");
        System.out.println("─".repeat(65));
        System.out.printf("  %-30s : %8d ms  (%5.1f%%)%n",
            "Phase 1 (1-itemsets)",
            phase1Time,
            totalTime > 0 ? (phase1Time * 100.0 / totalTime) : 0);

        System.out.printf("  %-30s : %8d ms  (%5.1f%%)%n",
            "Phase 2 (initialization)",
            phase2Time,
            totalTime > 0 ? (phase2Time * 100.0 / totalTime) : 0);

        System.out.printf("  %-30s : %8d ms  (%5.1f%%)%n",
            "Phase 3 (canonical mining)",
            phase3Time,
            totalTime > 0 ? (phase3Time * 100.0 / totalTime) : 0);

        System.out.println("─".repeat(65));
        System.out.printf("  %-30s : %8d ms%n", "Total execution time", totalTime);
        System.out.println();

        // Memory usage
        System.out.println("Memory Usage:");
        System.out.println("─".repeat(65));
        System.out.printf("  %-30s : %8.2f MB%n",
            "Memory consumed",
            memoryUsed / (1024.0 * 1024.0));
        System.out.println();

        // Throughput metrics
        System.out.println("Throughput Metrics:");
        System.out.println("─".repeat(65));

        if (totalTime > 0) {
            double transactionsPerSecond = (transactionCount * 1000.0) / totalTime;
            double patternsPerSecond = (patternCount * 1000.0) / totalTime;

            System.out.printf("  %-30s : %8.2f trans/sec%n",
                "Transaction throughput",
                transactionsPerSecond);

            System.out.printf("  %-30s : %8.2f patterns/sec%n",
                "Pattern discovery rate",
                patternsPerSecond);

            if (totalTime >= 1000) {
                System.out.printf("  %-30s : %8.3f sec/pattern%n",
                    "Average time per pattern",
                    totalTime / (1000.0 * patternCount));
            }
        }
        System.out.println();

        // Performance summary
        System.out.println("Performance Summary:");
        System.out.println("─".repeat(65));
        System.out.printf("  %-30s : %d%n", "Patterns found", patternCount);
        System.out.printf("  %-30s : %d%n", "Transactions processed", transactionCount);
        System.out.printf("  %-30s : %.2f ms%n", "Total time", (double) totalTime);

        // Performance rating based on time
        String rating;
        if (totalTime < 100) {
            rating = "Excellent (< 100ms)";
        } else if (totalTime < 1000) {
            rating = "Very Good (< 1 sec)";
        } else if (totalTime < 5000) {
            rating = "Good (< 5 sec)";
        } else if (totalTime < 30000) {
            rating = "Fair (< 30 sec)";
        } else {
            rating = "Slow (> 30 sec)";
        }
        System.out.printf("  %-30s : %s%n", "Performance rating", rating);

        System.out.println();
    }
}
