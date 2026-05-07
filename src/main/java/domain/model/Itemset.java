package domain.model;

import infrastructure.persistence.Vocabulary;

import java.util.ArrayList;
import java.util.BitSet;
import java.util.List;


/**
 * Itemset - Represents a set of items using BitSet for memory efficiency.
 *
 * In Frequent Itemset Mining, itemsets are the core data structure.
 * This class uses BitSet instead of Set<Integer> for:
 *   - Memory efficiency: 1 bit per item vs 32 bits for Integer object
 *   - Fast set operations: union, intersection via bitwise OR, AND
 *   - Cache-friendly: contiguous memory layout
 *
 * Example: Itemset {Bread, Milk, Butter} with indices {0, 2, 5}
 *          BitSet representation: 100101 (bits 0, 2, 5 are set)
 *
 * Immutability: Operations like union() return NEW Itemset, original unchanged.
 * After construction phase (calling add()), itemsets should be treated as immutable.
 *
 * This is the BASE CLASS for the itemset hierarchy:
 *   Itemset (just items)
 *       └── FrequentItemset (+ support, probability)
 *               └── CachedFrequentItemset (+ tidset)
 *
 * @author Dang Nguyen Le, Gia Huy Vo
 */
public class Itemset {

    /**
     * BitSet storing item indices.
     * Bit at position i is set (1) if item i is in this itemset.
     * Example: items = {0, 2, 5} means bits 0, 2, 5 are true.
     */
    private final BitSet items;

    /**
     * Reference to vocabulary for converting indices back to item names.
     * Shared across all itemsets to save memory.
     */
    private final Vocabulary vocab;

    /**
     * Public constructor - creates empty itemset.
     * Items can be added via add() during construction phase.
     *
     * @param vocab vocabulary for item name lookup
     */
    public Itemset(Vocabulary vocab) {
        // Create empty BitSet (all bits are false/0)
        this.items = new BitSet();

        // Store vocabulary reference for toStringWithCodec()
        this.vocab = vocab;
    }

    /**
     * Protected constructor - creates itemset from existing BitSet.
     * Used internally by union() and other operations, and by subclasses.
     *
     * @param items BitSet to clone (defensive copy)
     * @param vocab vocabulary reference
     */
    protected Itemset(BitSet items, Vocabulary vocab) {
        // Clone to ensure immutability - changes to original don't affect this
        this.items = (BitSet) items.clone();

        this.vocab = vocab;
    }

    /**
     * Protected copy constructor - creates copy of another itemset.
     * Useful for subclasses that need to copy base itemset data.
     *
     * @param other itemset to copy
     */
    protected Itemset(Itemset other) {
        this.items = (BitSet) other.items.clone();
        this.vocab = other.vocab;
    }

    /**
     * Static factory method - creates singleton itemset with one item.
     * Preferred over constructor + add() for single items.
     *
     * @param vocab vocabulary for item lookup
     * @param item  item index to include
     * @return new Itemset containing only the specified item
     */
    public static Itemset singleton(Vocabulary vocab, int item) {
        Itemset itemset = new Itemset(vocab);
        itemset.add(item);
        return itemset;
    }

    /**
     * Static factory method - creates itemset with multiple items.
     * Preferred over constructor + multiple add() calls.
     *
     * @param vocab vocabulary for item lookup
     * @param items item indices to include
     * @return new Itemset containing all specified items
     */
    public static Itemset of(Vocabulary vocab, int... items) {
        Itemset itemset = new Itemset(vocab);
        for (int item : items) {
            itemset.add(item);
        }
        return itemset;
    }

    /**
     * Add an item to this itemset.
     * Should only be called during construction phase.
     * After construction, treat itemsets as immutable.
     *
     * @param itemIndex index of item to add (from Vocabulary)
     * @throws IllegalArgumentException if index is negative
     */
    public void add(int itemIndex) {
        // Validate: BitSet doesn't accept negative indices
        if (itemIndex < 0) {
            throw new IllegalArgumentException(
                "Item index must be non-negative, got: " + itemIndex
            );
        }

        // Set bit at position itemIndex to true (1)
        // If already set, no change occurs
        items.set(itemIndex);
    }

    /**
     * Protected accessor for BitSet - allows subclasses to access underlying data.
     * Returns reference (not clone) for efficiency - subclasses must not modify.
     *
     * @return the underlying BitSet
     */
    protected BitSet getItemsBitSet() {
        return items;
    }

    /**
     * Get number of items in this itemset.
     *
     * @return count of items (number of set bits)
     */
    public int size() {
        // cardinality() counts number of bits set to true
        // More efficient than iterating: O(n/64) where n = highest bit
        return items.cardinality();
    }

    /**
     * Check if item is in this itemset.
     *
     * @param itemIndex index to check
     * @return true if item is present, false otherwise
     */
    public boolean contains(int itemIndex) {
        // get() returns true if bit at index is set, false otherwise
        // Returns false for negative indices (no exception)
        return items.get(itemIndex);
    }

    /**
     * Create union of this itemset with another.
     * Returns NEW itemset; originals unchanged (immutability).
     *
     * Set theory: A ∪ B = {x : x ∈ A or x ∈ B}
     * Example: {A,B} ∪ {B,C} = {A,B,C}
     *
     * @param other itemset to union with
     * @return new itemset containing all items from both
     */
    public Itemset union(Itemset other) {
        // Clone this itemset's bits (don't modify original)
        BitSet result = (BitSet) this.items.clone();

        // Bitwise OR: result[i] = this[i] OR other[i]
        // Sets bit to 1 if either itemset has it
        result.or(other.items);

        // Create new Itemset with combined bits
        return new Itemset(result, vocab);
    }

    /**
     * Get list of all item indices in this itemset.
     * Items are returned in ascending order (due to BitSet iteration).
     *
     * Used by mining algorithm to iterate through items.
     *
     * @return list of item indices in ascending order
     */
    public List<Integer> getItems() {
        List<Integer> result = new ArrayList<>();

        // nextSetBit(i) finds next bit that is true, starting from index i
        // Returns -1 if no more set bits exist
        // This efficiently skips over unset bits
        for (int i = items.nextSetBit(0); i >= 0; i = items.nextSetBit(i + 1)) {
            result.add(i);
        }

        return result;
    }

    /**
     * Get items as primitive int array (zero-copy, no boxing).
     *
     * <p><b>Performance:</b> Preferred over getItems() for iteration-heavy operations
     * as it avoids Integer boxing/unboxing overhead.</p>
     *
     * <p>Returns items in ascending order, same as getItems().</p>
     *
     * @return primitive array of item indices in ascending order
     */
    public int[] getItemsArray() {
        // Pre-allocate array of exact size (avoid resizing)
        int size = items.cardinality();
        int[] result = new int[size];

        int idx = 0;
        // Same iteration pattern as getItems(), but stores in primitive array
        for (int i = items.nextSetBit(0); i >= 0; i = items.nextSetBit(i + 1)) {
            result[idx++] = i;
        }

        return result;
    }

    /**
     * Get vocabulary associated with this itemset.
     *
     * @return vocabulary for item name lookup
     */
    public Vocabulary getVocabulary() {
        return vocab;
    }

    /**
     * Convert itemset to human-readable string using item names.
     *
     * Example: BitSet {0, 2, 5} -> "{Bread, Butter, Jam}"
     *
     * @return string representation with item names
     */
    public String toStringWithCodec() {
        // Handle empty itemset
        if (items.isEmpty()) return "{}";

        StringBuilder sb = new StringBuilder("{");
        boolean first = true;  // Track first item to handle comma placement

        // Iterate through all set bits
        for (int i = items.nextSetBit(0); i >= 0; i = items.nextSetBit(i + 1)) {
            // Add comma before all items except first
            if (!first) sb.append(", ");

            // Look up item name from vocabulary and append
            sb.append(vocab.getItem(i));

            first = false;
        }

        sb.append("}");
        return sb.toString();
    }

    /**
     * Convert itemset to string representation.
     * Delegates to toStringWithCodec() for human-readable output.
     *
     * @return string representation with item names
     */
    @Override
    public String toString() {
        return toStringWithCodec();
    }

    /**
     * Check equality based on items contained (not vocabulary reference).
     * Two itemsets are equal if they contain exactly the same items.
     *
     * @param obj object to compare
     * @return true if same items, false otherwise
     */
    @Override
    public boolean equals(Object obj) {
        // Same reference check (optimization)
        if (this == obj) return true;

        // Type check
        if (!(obj instanceof Itemset)) return false;

        Itemset other = (Itemset) obj;

        // BitSet.equals() compares bit by bit
        return this.items.equals(other.items);
    }

    /**
     * Hash code based on items for use in HashMap/HashSet.
     * Consistent with equals(): equal itemsets have same hash.
     *
     * @return hash code
     */
    @Override
    public int hashCode() {
        // BitSet.hashCode() is based on set bits
        return items.hashCode();
    }
}