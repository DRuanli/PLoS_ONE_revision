package infrastructure.persistence;

import java.util.Map;
import java.util.HashMap;
import java.util.List;
import java.util.ArrayList;

/**
 * Vocabulary - Bidirectional mapping between item names and integer indices.
 *
 * Purpose: In Frequent Itemset Mining, representing items as integers instead of
 * strings provides significant performance benefits:
 *   - Faster comparison: integer comparison vs string comparison
 *   - Enables BitSet representation for itemsets
 *
 * Data Structure:
 *   - Forward mapping:  "Bread" -> 0, "Milk" -> 1  (HashMap)
 *   - Reverse mapping:  0 -> "Bread", 1 -> "Milk"  (ArrayList)
 *
 *
 * @author Dang Nguyen Le, Gia Huy Vo
 */
public class Vocabulary {
    /**
     * Forward mapping: item name -> index
     * Uses HashMap for O(1) average lookup time.
     * Example: {"Bread" -> 0, "Milk" -> 1, "Butter" -> 2}
     */
    private final Map<String, Integer> itemToIndex;

    /**
     * Reverse mapping: index -> item name
     * Uses ArrayList for O(1) access by index.
     * Example: [0: "Bread", 1: "Milk", 2: "Butter"]
     */
    private final List<String> indexToItem;

    /**
     * Default constructor - creates an empty vocabulary.
     * Items will be added dynamically when loading the database.
     */
    public Vocabulary() {
        // Initialize HashMap for name-to-index mapping
        this.itemToIndex = new HashMap<>();

        // Initialize ArrayList for index-to-name mapping
        this.indexToItem = new ArrayList<>();
    }


    /**
     * Get existing index or create new index for an item.
     *
     * This method implements the Lazy Initialization pattern:
     *   - If item exists: return its index (no modification)
     *   - If item is new: assign next available index, store mapping, return index
     *
     * Example:
     *   vocabulary.getOrCreateIndex("Bread")  -> 0 (first item)
     *   vocabulary.getOrCreateIndex("Milk")   -> 1 (second item)
     *   vocabulary.getOrCreateIndex("Bread")  -> 0 (already exists, same index)
     *
     * @param item the item name to look up or register
     * @return the integer index associated with this item
     */
    public int getOrCreateIndex(String item) {
        // computeIfAbsent: atomic operation that either:
        //   1. Returns existing value if key present, OR
        //   2. Computes new value, stores it, and returns it
        return itemToIndex.computeIfAbsent(item, k -> {
            // 'k' is the key (same as 'item') passed to this lambda

            // New index = current size (indices are 0, 1, 2, ...)
            // If list has 3 items [0,1,2], next index is 3
            int index = indexToItem.size();

            // Add item name to reverse mapping list
            // After this: indexToItem.get(index) returns 'item'
            indexToItem.add(k);

            // Return index to be stored in itemToIndex map
            // After this: itemToIndex.get(item) returns 'index'
            return index;
        });
    }

    /**
     * Get item name by its index (reverse lookup).
     *
     * Used when displaying mining results to convert internal
     * representation back to human-readable item names.
     *
     * Example: getItem(0) -> "Bread"
     *
     * @param index the integer index to look up
     * @return the item name at that index
     * @throws IndexOutOfBoundsException if index is out of range [0, size-1]
     */
    public String getItem(int index) {
        // Direct array access - O(1) time complexity
        return indexToItem.get(index);
    }

    /**
     * Get the total number of unique items in vocabulary.
     *
     * Used in mining algorithm to:
     *   - Determine BitSet size for itemset representation
     *   - Iterate through all items in Phase 1
     *
     * @return number of registered items
     */
    public int size() {
        // Both maps have same size, return either one
        return indexToItem.size();
    }
}
