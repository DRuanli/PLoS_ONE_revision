package infrastructure.persistence;

import domain.model.Tidset;
import domain.model.Itemset;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;

/**
 * UncertainDatabase - Stores and manages uncertain transaction database.
 *
 * In uncertain databases, each item in a transaction has an existence probability:
 *   Transaction T1: {(Bread, 0.9), (Milk, 0.7), (Butter, 0.5)}
 *   Meaning: Bread appears with 90% probability, Milk with 70%, etc.
 *
 * This class supports TWO representations:
 *
 * 1. HORIZONTAL (during loading):
 *    Transaction -> {(item, probability)}
 *    T1 -> {(A, 0.9), (B, 0.7)}
 *    T2 -> {(A, 0.6), (C, 0.8)}
 *
 * 2. VERTICAL (for mining):
 *    Item -> {(transaction, probability)}
 *    A -> {(T1, 0.9), (T2, 0.6)}
 *    B -> {(T1, 0.7)}
 *    C -> {(T2, 0.8)}
 *
 * Vertical representation enables efficient tidset intersection for itemset mining.
 *
 * File Format:
 *   <tid> <item1>:<prob1> <item2>:<prob2> ...
 *   Example:
 *     1 Bread:0.9 Milk:0.7 Butter:0.5
 *     2 Bread:0.6 Cheese:0.8
 *
 * @author Dang Nguyen Le, Gia Huy Vo
 */
public class UncertainDatabase {

    /**
     * Horizontal representation: transaction ID -> {item index -> probability}
     * Used during database loading phase.
     *
     * Example: {0 -> {0 -> 0.9, 1 -> 0.7}, 1 -> {0 -> 0.6, 2 -> 0.8}}
     *          Transaction 0 has item 0 (prob 0.9) and item 1 (prob 0.7)
     */
    private final Map<Integer, Map<Integer, Double>> transactions;

    /**
     * Vocabulary for item name <-> index mapping.
     * Shared with Itemset objects for consistent encoding.
     */
    private final Vocabulary vocab;

    /**
     * Vertical representation: item index -> Tidset
     * Built from horizontal representation for efficient mining.
     * Null until buildVerticalDatabase() is called.
     */
    private Map<Integer, Tidset> verticalDB;

    /**
     * Flag indicating if vertical database has been built.
     * Once true, no more transactions can be added (immutability).
     */
    private boolean verticalBuilt;

    /**
     * Optional name for this database (e.g., "chess", "mushroom").
     * Used for logging and experiment tracking.
     */
    private String name;

    /**
     * Constructor - creates empty database with given vocabulary.
     *
     * @param vocab vocabulary for item encoding
     */
    public UncertainDatabase(Vocabulary vocab) {
        // Initialize horizontal storage
        this.transactions = new HashMap<>();

        // Store vocabulary reference
        this.vocab = vocab;

        // Vertical DB not built yet
        this.verticalDB = null;
        this.verticalBuilt = false;
    }

    /**
     * Add a transaction to the database.
     *
     * Must be called BEFORE buildVerticalDatabase().
     * Database becomes immutable after vertical DB is built.
     *
     * @param tid   transaction ID (unique identifier)
     * @param items map of item index -> probability
     * @throws IllegalStateException if vertical DB already built
     * @throws IllegalArgumentException if probability is invalid
     */
    public void addTransaction(int tid, Map<Integer, Double> items) {
        // Check immutability constraint
        if (verticalBuilt) {
            throw new IllegalStateException(
                "Cannot add transactions after vertical database is built. " +
                "Database is immutable once buildVerticalDatabase() is called."
            );
        }

        // Validate all probabilities
        for (Map.Entry<Integer, Double> entry : items.entrySet()) {
            double prob = entry.getValue();

            // Check for NaN/Infinite (invalid floating point values)
            if (Double.isNaN(prob) || Double.isInfinite(prob)) {
                throw new IllegalArgumentException("Invalid probability (NaN/Infinite)");
            }

            // Check probability range [0, 1]
            if (prob < 0.0 || prob > 1.0) {
                throw new IllegalArgumentException("Probability must be in [0.0, 1.0]");
            }
        }

        // Store defensive copy of items map
        transactions.put(tid, new HashMap<>(items));
    }

    /**
     * Get number of transactions in database.
     *
     * @return transaction count
     */
    public int size() {
        return transactions.size();
    }

    /**
     * Get the name of this database.
     *
     * @return database name, or "unnamed" if not set
     */
    public String getName() {
        return name != null ? name : "unnamed";
    }

    /**
     * Set the name of this database.
     *
     * @param name database name
     */
    public void setName(String name) {
        this.name = name;
    }

    /**
     * Get vocabulary for item encoding/decoding.
     *
     * @return vocabulary instance
     */
    public Vocabulary getVocabulary() {
        return vocab;
    }

    /**
     * Build vertical database representation from horizontal transactions.
     *
     * Transforms:
     *   Horizontal: T1 -> {A:0.9, B:0.7}
     *   Vertical:   A -> {(T1, 0.9)}, B -> {(T1, 0.7)}
     *
     * After this call:
     *   - Database becomes immutable (no more addTransaction)
     *   - getTidset() methods become available
     *   - Tidsets are sorted by transaction ID for efficient intersection
     *
     * Time Complexity: O(N × M) where N = transactions, M = avg items per transaction
     *
     * @throws IllegalStateException if already built or no transactions
     */
    public void buildVerticalDatabase() {
        // Prevent rebuilding
        if (verticalBuilt) {
            throw new IllegalStateException(
                "Vertical database already built. Cannot rebuild."
            );
        }

        // Validate non-empty database
        if (transactions.isEmpty()) {
            throw new IllegalStateException(
                "Cannot build vertical database with no transactions"
            );
        }

        // Initialize vertical structure
        verticalDB = new HashMap<>();

        // Transform horizontal -> vertical
        // For each transaction...
        for (Map.Entry<Integer, Map<Integer, Double>> entry : transactions.entrySet()) {
            int tid = entry.getKey();                    // Transaction ID
            Map<Integer, Double> transaction = entry.getValue();  // Items in transaction

            // For each item in transaction...
            for (Map.Entry<Integer, Double> itemEntry : transaction.entrySet()) {
                int item = itemEntry.getKey();           // Item index
                double prob = itemEntry.getValue();      // Existence probability

                // Get or create tidset for this item
                // computeIfAbsent: if item not in map, create new Tidset
                Tidset tidset = verticalDB.computeIfAbsent(item, k -> new Tidset());

                // Add (tid, prob) to item's tidset
                tidset.add(tid, prob);
            }
        }

        // Sort all tidsets by transaction ID
        // Required for merge-join intersection algorithm in Tidset.intersect()
        for (Tidset tidset : verticalDB.values()) {
            List<Tidset.TIDProb> entries = tidset.getEntries();
            Collections.sort(entries);  // TIDProb implements Comparable<TIDProb>
        }

        // Mark as built (enables getTidset, disables addTransaction)
        verticalBuilt = true;
    }

    /**
     * Get tidset for a single item.
     *
     * @param item item index
     * @return tidset containing all transactions with this item
     * @throws IllegalStateException if vertical DB not built
     */
    public Tidset getTidset(int item) {
        // Ensure vertical DB is available
        if (!verticalBuilt || verticalDB == null) {
            throw new IllegalStateException(
                "Vertical database not built. Call buildVerticalDatabase() first."
            );
        }

        // Lookup tidset, return empty if item not found
        Tidset tidset = verticalDB.get(item);
        return tidset != null ? tidset : new Tidset();
    }

    /**
     * Get tidset for an itemset by intersecting individual item tidsets.
     *
     * Algorithm:
     *   1. Sort items by tidset size (ascending) - optimization
     *   2. Start with smallest tidset
     *   3. Intersect with remaining tidsets in order
     *   4. Early termination if result becomes empty
     *
     * Why sort by size?
     *   - Intersection result size ≤ min(input sizes)
     *   - Processing smallest first keeps intermediate results small
     *   - Reduces total comparisons in merge-join
     *
     * Example: tidset({A,B,C}) = tidset(A) ∩ tidset(B) ∩ tidset(C)
     *
     * @param itemset the itemset to get tidset for
     * @return tidset containing transactions with ALL items in itemset
     * @throws IllegalStateException if vertical DB not built
     */
    public Tidset getTidset(Itemset itemset) {
        // Ensure vertical DB is available
        if (!verticalBuilt || verticalDB == null) {
            throw new IllegalStateException(
                "Vertical database not built. Call buildVerticalDatabase() first."
            );
        }

        // Get items as primitive array to avoid boxing overhead
        int[] items = itemset.getItemsArray();

        // Handle empty itemset
        if (items.length == 0) return new Tidset();

        // Handle singleton itemset (no intersection needed)
        if (items.length == 1) {
            return getTidset(items[0]);
        }

        // OPTIMIZATION: Sort items by tidset size (ascending)
        // Smallest tidset first minimizes intermediate result sizes
        // Need to box into List for sorting (unavoidable here)
        List<Integer> sortedItems = new ArrayList<>(items.length);
        for (int item : items) {
            sortedItems.add(item);
        }
        sortedItems.sort(Comparator.comparingInt(item -> {
            Tidset t = verticalDB.get(item);
            return t != null ? t.size() : 0;
        }));

        // Start with smallest tidset
        Tidset result = getTidset(sortedItems.get(0));

        // Intersect with remaining tidsets
        for (int i = 1; i < sortedItems.size(); i++) {
            Tidset itemTidset = getTidset(sortedItems.get(i));
            result = result.intersect(itemTidset);

            // Early termination: empty intersection stays empty
            if (result.isEmpty()) break;
        }

        return result;
    }

    /**
     * Load uncertain database from file.
     *
     * File format:
     *   [optional header: num_transactions num_items]
     *   <tid> <item1>:<prob1> <item2>:<prob2> ...
     *
     * Example file:
     *   1 Bread:0.9 Milk:0.7
     *   2 Bread:0.6 Cheese:0.8 Butter:0.5
     *   3 Milk:0.9 Cheese:0.7
     *
     * Note: Automatically calls buildVerticalDatabase() after loading.
     *
     * @param filename path to database file
     * @return loaded and prepared database
     * @throws IOException if file cannot be read
     */
    public static UncertainDatabase loadFromFile(String filename) throws IOException {
        // Create vocabulary and database
        Vocabulary vocab = new Vocabulary();
        UncertainDatabase db = new UncertainDatabase(vocab);

        // Read file using try-with-resources (auto-close)
        try (BufferedReader br = new BufferedReader(new FileReader(filename))) {
            // Read first line
            String firstLine = br.readLine();
            if (firstLine == null) throw new IOException("Empty file");

            // Check if first line is header (two integers: num_trans num_items)
            String[] parts = firstLine.trim().split("\\s+");
            boolean hasHeader = false;

            if (parts.length == 2) {
                try {
                    // Try parsing as two integers
                    Integer.parseInt(parts[0]);
                    Integer.parseInt(parts[1]);
                    hasHeader = true;  // Success: it's a header
                } catch (NumberFormatException e) {
                    hasHeader = false;  // Not integers: it's data
                }
            }

            // If first line is data (not header), process it
            if (!hasHeader) processLine(firstLine, db, vocab);

            // Process remaining lines
            String line;
            while ((line = br.readLine()) != null) {
                processLine(line, db, vocab);
            }
        }

        // Build vertical database for efficient mining
        db.buildVerticalDatabase();

        return db;
    }

    /**
     * Process a single line from input file.
     *
     * Format: <tid> <item1>:<prob1> <item2>:<prob2> ...
     *
     * @param line raw line from file
     * @param db   database to add transaction to
     * @param vocab vocabulary for item encoding
     */
    private static void processLine(String line, UncertainDatabase db, Vocabulary vocab) {
        // Trim whitespace
        line = line.trim();

        // Skip empty lines
        if (line.isEmpty()) return;

        // Split by whitespace
        String[] parts = line.split("\\s+");

        // Need at least tid and one item
        if (parts.length < 2) return;

        // First part is transaction ID
        int tid = Integer.parseInt(parts[0]);

        // Build transaction map
        Map<Integer, Double> transaction = new HashMap<>();

        // Process each item:prob pair
        for (int i = 1; i < parts.length; i++) {
            // Split "item:prob" by colon
            String[] itemProb = parts[i].split(":");

            // Validate format
            if (itemProb.length != 2) continue;

            // Extract item name and probability
            String itemName = itemProb[0];
            double prob = Double.parseDouble(itemProb[1]);

            // Get/create item index from vocabulary
            int itemIndex = vocab.getOrCreateIndex(itemName);

            // Add to transaction
            transaction.put(itemIndex, prob);
        }

        // Add transaction to database
        db.addTransaction(tid, transaction);
    }
}
