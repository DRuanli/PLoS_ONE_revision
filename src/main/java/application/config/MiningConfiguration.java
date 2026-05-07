package application.config;

/**
 * Immutable configuration object for Top-K Uncertain Frequent Itemset mining parameters.
 * Uses the Builder pattern to construct instances with required and optional parameters.
 *
 * <p>This configuration encapsulates all necessary parameters for running the mining algorithms,
 * including database location, threshold values, algorithm selection, and output preferences.</p>
 *
 * @see Builder
 */
public class MiningConfiguration {
    /** Path to the uncertain database file to be mined */
    private final String databasePath;

    /** Minimum expected support threshold (tau), must be in range (0, 1] */
    private final double tau;

    /** Number of top frequent itemsets to find (K), must be positive */
    private final int k;

    /** Name of the mining algorithm to use (e.g., "TUFCI", "TUFCI_BFS", "TUFCI_DFS") */
    private final String algorithmName;

    /** Flag to enable verbose output during mining process */
    private final boolean verbose;

    //private final PruningConfiguration pruningConfig;

    /**
     * Private constructor called by Builder to create an immutable configuration instance.
     *
     * @param builder the builder containing validated configuration parameters
     */
    private MiningConfiguration(Builder builder) {
        this.databasePath = builder.databasePath;
        this.tau = builder.tau;
        this.k = builder.k;
        this.algorithmName = builder.algorithmName;
        this.verbose = builder.verbose;
        //this.pruningConfig = builder.pruningConfig;
    }

    /**
     * Returns the path to the uncertain database file.
     *
     * @return the database file path
     */
    public String getDatabasePath() { return databasePath; }

    /**
     * Returns the minimum expected support threshold (tau).
     *
     * @return the tau value in range (0, 1]
     */
    public double getTau() { return tau; }

    /**
     * Returns the number of top frequent itemsets to find.
     *
     * @return the K value (positive integer)
     */
    public int getK() { return k; }

    /**
     * Returns the name of the mining algorithm to use.
     *
     * @return the algorithm name (e.g., "TUFCI", "TUFCI_BFS", "TUFCI_DFS")
     */
    public String getAlgorithmName() { return algorithmName; }

    /**
     * Returns whether verbose output is enabled.
     *
     * @return true if verbose mode is enabled, false otherwise
     */
    public boolean isVerbose() { return verbose; }

    //public PruningConfiguration getPruningConfig() { return pruningConfig; }

    /**
     * Returns a string representation of this configuration.
     *
     * @return a formatted string with key configuration parameters
     */
    @Override
    public String toString() {
        return String.format("MiningConfig[db=%s, tau=%.2f, k=%d, algo=%s]",
                           databasePath, tau, k, algorithmName);
    }

    /**
     * Builder class for constructing MiningConfiguration instances.
     * Provides a fluent interface for setting configuration parameters with sensible defaults.
     *
     * <p>Example usage:</p>
     * <pre>
     * MiningConfiguration config = new MiningConfiguration.Builder()
     *     .databasePath("data/database.txt")
     *     .tau(0.8)
     *     .k(20)
     *     .algorithmName("TUFCI_BFS")
     *     .verbose(true)
     *     .build();
     * </pre>
     */
    public static class Builder {
        /** Path to the database file (required, no default) */
        private String databasePath;

        /** Default minimum expected support threshold */
        private double tau = 0.7;

        /** Default number of top itemsets to find */
        private int k = 10;

        /** Default algorithm to use */
        private String algorithmName = "TUFCI";

        /** Default verbose mode setting */
        private boolean verbose = false;

        //private PruningConfiguration pruningConfig = PruningConfiguration.allEnabled();

        /**
         * Sets the path to the uncertain database file to be mined.
         *
         * @param path the database file path (required)
         * @return this builder for method chaining
         */
        public Builder databasePath(String path) {
            this.databasePath = path;
            return this;
        }

        /**
         * Sets the minimum expected support threshold (tau).
         *
         * @param tau the threshold value, must be in range (0, 1]
         * @return this builder for method chaining
         */
        public Builder tau(double tau) {
            this.tau = tau;
            return this;
        }

        /**
         * Sets the number of top frequent itemsets to find.
         *
         * @param k the number of itemsets (must be positive)
         * @return this builder for method chaining
         */
        public Builder k(int k) {
            this.k = k;
            return this;
        }

        /**
         * Sets the name of the mining algorithm to use.
         *
         * @param name the algorithm name (e.g., "TUFCI", "TUFCI_BFS", "TUFCI_DFS")
         * @return this builder for method chaining
         */
        public Builder algorithmName(String name) {
            this.algorithmName = name;
            return this;
        }

        /**
         * Sets whether to enable verbose output during mining.
         *
         * @param verbose true to enable verbose mode, false otherwise
         * @return this builder for method chaining
         */
        public Builder verbose(boolean verbose) {
            this.verbose = verbose;
            return this;
        }

        //public Builder pruningConfig(PruningConfiguration config) {
        //    this.pruningConfig = config;
        //    return this;
        //}

        /**
         * Validates all configuration parameters and builds an immutable MiningConfiguration instance.
         *
         * <p>Validation rules:</p>
         * <ul>
         *   <li>Database path must be non-null and non-empty</li>
         *   <li>Tau must be in range (0, 1]</li>
         *   <li>K must be positive</li>
         * </ul>
         *
         * @return a new immutable MiningConfiguration instance
         * @throws IllegalArgumentException if any validation check fails
         */
        public MiningConfiguration build() {
            if (databasePath == null || databasePath.isEmpty()) {
                throw new IllegalArgumentException("Database path is required");
            }
            if (tau <= 0 || tau > 1) {
                throw new IllegalArgumentException("Tau must be in (0, 1]");
            }
            if (k <= 0) {
                throw new IllegalArgumentException("K must be positive");
            }
            return new MiningConfiguration(this);
        }
    }
}