package presentation;

import domain.observer.PhaseTimingObserver;
import infrastructure.factory.MinerFactory;
import domain.model.FrequentItemset;
import infrastructure.persistence.UncertainDatabase;
import presentation.Usage;
import domain.mining.TUFCI;

import java.io.IOException;
import java.util.List;

public class Main {
    public static void main(String[] args) throws IOException {
        if (args.length < 1){
            Usage.printUsage();
        }

        String dbFile = args[0];
        double tau = args.length > 1 ? Double.parseDouble(args[1]) : 0.7;
        int k = args.length > 2 ? Integer.parseInt(args[2]) : 5;

        // Print configuration
        System.out.println("╔═══════════════════════════════════════════════════════════╗");
        System.out.println("║          TUFCI: Top-K Uncertain Frequent Closed           ║");
        System.out.println("║              Itemset Mining Algorithm                     ║");
        System.out.println("╚═══════════════════════════════════════════════════════════╝");
        System.out.println();
        System.out.println("Configuration:");
        System.out.println("  Database file     : " + dbFile);
        System.out.println("  Tau (threshold)   : " + tau);
        System.out.println("  K (top patterns)  : " + k);
        System.out.println();

        System.out.println("Loading database...");
        UncertainDatabase database = UncertainDatabase.loadFromFile(dbFile);

        System.out.println("  Transactions : " + database.size());
        System.out.println("  Vocabulary   : " + database.getVocabulary().size() + " unique items");
        System.out.println();


        System.out.println("Creating TUFCI miner...");
        TUFCI miner = MinerFactory.createMiner(database, tau, k);

        PhaseTimingObserver observer = new PhaseTimingObserver();
        System.out.println("  Algorithm: TUFCI");
        System.out.println();

        System.out.println("Starting mining process...");
        System.out.println("─".repeat(65));

        // Get memory usage before mining
        Runtime runtime = Runtime.getRuntime();
        runtime.gc(); // Suggest garbage collection for more accurate measurement
        long memoryBefore = runtime.totalMemory() - runtime.freeMemory();

        // Record start time to measure performance
        long startTime = System.nanoTime();

        // Execute the mining algorithm
        List<FrequentItemset> results = miner.mine();

        long endTime = System.nanoTime();
        long executionTime = (endTime - startTime) / 1_000_000; // Convert to ms
        long memoryAfter = runtime.totalMemory() - runtime.freeMemory();
        long memoryUsed = memoryAfter - memoryBefore;

        System.out.println("─".repeat(65));
        System.out.println("Mining completed!");
        System.out.println();

        // ==================== Step 5: Display Performance Metrics ====================

        Usage.printPerformanceMetrics(observer, executionTime, memoryUsed,
                               database.size(), results.size());

        // ==================== Step 6: Display Results ====================

        Usage.printResults(results, k);

    }

}
