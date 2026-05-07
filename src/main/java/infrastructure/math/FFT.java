package infrastructure.math;

/**
 * FFT - Fast Fourier Transform implementation.
 *
 * Uses Cooley-Tukey radix-2 decimation-in-time algorithm.
 *
 * ═══════════════════════════════════════════════════════════════════════════════
 * WHAT IS FFT?
 * ═══════════════════════════════════════════════════════════════════════════════
 *
 * FFT efficiently computes the Discrete Fourier Transform (DFT):
 *   X[k] = Σ(n=0 to N-1) x[n] · e^(-2πikn/N)
 *
 * DFT converts signal from TIME domain to FREQUENCY domain.
 *
 * Naive DFT: O(n²) - compute each of n outputs using n multiplications
 * FFT:       O(n log n) - divide-and-conquer reduces work dramatically
 *
 * ═══════════════════════════════════════════════════════════════════════════════
 * WHY FFT FOR POLYNOMIAL MULTIPLICATION?
 * ═══════════════════════════════════════════════════════════════════════════════
 *
 * Convolution Theorem:
 *   Convolution in time domain = Pointwise multiplication in frequency domain
 *
 * Polynomial multiplication IS convolution of coefficients!
 *
 * Algorithm:
 *   1. FFT(polynomial A) → frequency domain A'
 *   2. FFT(polynomial B) → frequency domain B'
 *   3. Pointwise multiply: C'[i] = A'[i] × B'[i]
 *   4. IFFT(C') → result polynomial C
 *
 * Complexity:
 *   - Naive multiplication: O(n²)
 *   - FFT multiplication: O(n log n)
 *
 * ═══════════════════════════════════════════════════════════════════════════════
 * COOLEY-TUKEY ALGORITHM
 * ═══════════════════════════════════════════════════════════════════════════════
 *
 * Key insight: DFT of size n can be computed from two DFTs of size n/2.
 *
 * Split input into EVEN and ODD indices:
 *   even = [x[0], x[2], x[4], ...]
 *   odd  = [x[1], x[3], x[5], ...]
 *
 * Combine using "butterfly" operation:
 *   X[k]     = even[k] + w_k · odd[k]
 *   X[k+n/2] = even[k] - w_k · odd[k]
 *
 * Where w_k = e^(-2πik/n) is the "twiddle factor".
 *
 * Time: T(n) = 2T(n/2) + O(n) → O(n log n)
 *
 * @author Dang Nguyen Le, Gia Huy Vo
 *
 */
public class FFT {

    /**
     * Compute FFT of complex array using Cooley-Tukey radix-2 algorithm.
     *
     * ═══════════════════════════════════════════════════════════════════════
     * ALGORITHM STEPS:
     * ═══════════════════════════════════════════════════════════════════════
     *
     * 1. BASE CASE: If n = 1, return input (DFT of single element is itself)
     *
     * 2. DIVIDE: Split into even and odd indexed elements
     *    even = [x[0], x[2], x[4], ...]
     *    odd  = [x[1], x[3], x[5], ...]
     *
     * 3. CONQUER: Recursively compute FFT of each half
     *    evenFFT = FFT(even)
     *    oddFFT  = FFT(odd)
     *
     * 4. COMBINE: Use butterfly operation with twiddle factors
     *    For k = 0 to n/2 - 1:
     *      w_k = e^(-2πik/n) = cos(-2πk/n) + i·sin(-2πk/n)
     *      t = w_k × oddFFT[k]
     *      result[k]     = evenFFT[k] + t
     *      result[k+n/2] = evenFFT[k] - t
     *
     * ═══════════════════════════════════════════════════════════════════════
     *
     * Time Complexity: O(n log n)
     * Space Complexity: O(n) for recursion stack and temporary arrays
     *
     * @param x input array (length MUST be power of 2)
     * @return FFT of x
     * @throws IllegalArgumentException if length is not power of 2
     */
    public static Complex[] fft(Complex[] x) {
        int n = x.length;

        // ─────────────────────────────────────────────────────────────
        // BASE CASE: DFT of single element is the element itself
        // ─────────────────────────────────────────────────────────────
        if (n == 1) {
            return new Complex[] { x[0] };
        }

        // ─────────────────────────────────────────────────────────────
        // VALIDATION: Length must be power of 2 for radix-2 algorithm
        // ─────────────────────────────────────────────────────────────
        if (n % 2 != 0) {
            throw new IllegalArgumentException("n must be a power of 2");
        }

        // ─────────────────────────────────────────────────────────────
        // DIVIDE: Split into even and odd indexed elements
        // even = x[0], x[2], x[4], ...
        // odd  = x[1], x[3], x[5], ...
        // ─────────────────────────────────────────────────────────────
        Complex[] even = new Complex[n / 2];
        Complex[] odd = new Complex[n / 2];

        for (int k = 0; k < n / 2; k++) {
            even[k] = x[2 * k];      // Even indices: 0, 2, 4, ...
            odd[k] = x[2 * k + 1];   // Odd indices: 1, 3, 5, ...
        }

        // ─────────────────────────────────────────────────────────────
        // CONQUER: Recursively compute FFT of each half
        // ─────────────────────────────────────────────────────────────
        Complex[] evenFFT = fft(even);
        Complex[] oddFFT = fft(odd);

        // ─────────────────────────────────────────────────────────────
        // COMBINE: Butterfly operation with twiddle factors
        //
        // Twiddle factor w_k = e^(-2πik/n)
        // Using Euler's formula: e^(iθ) = cos(θ) + i·sin(θ)
        // So w_k = cos(-2πk/n) + i·sin(-2πk/n)
        //
        // Butterfly:
        //   result[k]     = evenFFT[k] + w_k × oddFFT[k]
        //   result[k+n/2] = evenFFT[k] - w_k × oddFFT[k]
        // ─────────────────────────────────────────────────────────────
        Complex[] result = new Complex[n];

        for (int k = 0; k < n / 2; k++) {
            // Compute twiddle factor: w_k = e^(-2πik/n)
            double angle = -2.0 * Math.PI * k / n;
            Complex wk = new Complex(Math.cos(angle), Math.sin(angle));

            // Multiply twiddle factor with odd FFT result
            Complex t = wk.times(oddFFT[k]);

            // Butterfly operation
            result[k] = evenFFT[k].plus(t);           // First half
            result[k + n / 2] = evenFFT[k].minus(t);  // Second half
        }

        return result;
    }

    /**
     * Compute inverse FFT of complex array.
     *
     * IFFT reverses FFT: IFFT(FFT(x)) = x
     *
     * Algorithm using conjugate trick:
     *   IFFT(x) = conjugate(FFT(conjugate(x))) / n
     *
     * Why this works:
     *   - IFFT uses e^(+2πik/n) instead of e^(-2πik/n)
     *   - Conjugate flips the sign of imaginary part
     *   - Two conjugates cancel out, but effectively flip the sign in exponent
     *   - Division by n normalizes the result
     *
     * @param x input array (length must be power of 2)
     * @return inverse FFT of x
     */
    public static Complex[] ifft(Complex[] x) {
        int n = x.length;

        // ─────────────────────────────────────────────────────────────
        // STEP 1: Take conjugate of input
        // ─────────────────────────────────────────────────────────────
        Complex[] y = new Complex[n];
        for (int i = 0; i < n; i++) {
            y[i] = x[i].conjugate();
        }

        // ─────────────────────────────────────────────────────────────
        // STEP 2: Compute forward FFT
        // ─────────────────────────────────────────────────────────────
        y = fft(y);

        // ─────────────────────────────────────────────────────────────
        // STEP 3: Take conjugate again and divide by n
        // ─────────────────────────────────────────────────────────────
        for (int i = 0; i < n; i++) {
            y[i] = y[i].conjugate().times(1.0 / n);
        }

        return y;
    }

    /**
     * Multiply two polynomials using FFT-based convolution.
     *
     * ═══════════════════════════════════════════════════════════════════════
     * POLYNOMIAL MULTIPLICATION VIA CONVOLUTION THEOREM
     * ═══════════════════════════════════════════════════════════════════════
     *
     * Given polynomials:
     *   A(x) = a[0] + a[1]x + a[2]x² + ...
     *   B(x) = b[0] + b[1]x + b[2]x² + ...
     *
     * Product C(x) = A(x) × B(x) has coefficients:
     *   c[k] = Σ(i=0 to k) a[i] × b[k-i]  (convolution!)
     *
     * Convolution Theorem:
     *   FFT(A * B) = FFT(A) × FFT(B)  (pointwise multiplication)
     *
     * Algorithm:
     *   1. Pad both polynomials to size 2^k ≥ deg(A) + deg(B) + 1
     *   2. FFT both padded polynomials
     *   3. Pointwise multiply in frequency domain
     *   4. IFFT to get result coefficients
     *
     * Example: (1 + 2x) × (3 + 4x)
     *   A = [1, 2], B = [3, 4]
     *   Pad to size 4: A = [1,2,0,0], B = [3,4,0,0]
     *   FFT(A) × FFT(B) then IFFT
     *   Result: [3, 10, 8] → 3 + 10x + 8x²
     *
     * ═══════════════════════════════════════════════════════════════════════
     *
     * Time Complexity: O(n log n) where n = |a| + |b|
     * Space Complexity: O(n)
     *
     * @param a coefficients of first polynomial [a0, a1, a2, ...]
     * @param b coefficients of second polynomial [b0, b1, b2, ...]
     * @return coefficients of product polynomial
     */
    public static double[] multiplyPolynomials(double[] a, double[] b) {
        // Result polynomial degree = deg(a) + deg(b) = (|a|-1) + (|b|-1)
        // So result has |a| + |b| - 1 coefficients
        int resultDegree = a.length + b.length - 1;

        // ─────────────────────────────────────────────────────────────
        // STEP 1: Find next power of 2 for FFT (padding)
        // FFT requires input size to be power of 2
        // ─────────────────────────────────────────────────────────────
        int fftSize = nextPowerOf2(resultDegree);

        // ─────────────────────────────────────────────────────────────
        // STEP 2: Convert to complex arrays with zero padding
        // Padding ensures arrays are same size and power of 2
        // ─────────────────────────────────────────────────────────────
        Complex[] aComplex = new Complex[fftSize];
        Complex[] bComplex = new Complex[fftSize];

        for (int i = 0; i < fftSize; i++) {
            // Copy coefficient if exists, else pad with zero
            aComplex[i] = i < a.length ? new Complex(a[i]) : new Complex(0);
            bComplex[i] = i < b.length ? new Complex(b[i]) : new Complex(0);
        }

        // ─────────────────────────────────────────────────────────────
        // STEP 3: Compute FFT of both polynomials
        // Transforms from coefficient representation to point-value
        // ─────────────────────────────────────────────────────────────
        Complex[] aFFT = fft(aComplex);
        Complex[] bFFT = fft(bComplex);

        // ─────────────────────────────────────────────────────────────
        // STEP 4: Pointwise multiplication in frequency domain
        // This is where the magic happens!
        // Convolution becomes simple multiplication
        // ─────────────────────────────────────────────────────────────
        Complex[] productFFT = new Complex[fftSize];

        for (int i = 0; i < fftSize; i++) {
            productFFT[i] = aFFT[i].times(bFFT[i]);
        }

        // ─────────────────────────────────────────────────────────────
        // STEP 5: Inverse FFT to get result coefficients
        // Transforms back from point-value to coefficient representation
        // ─────────────────────────────────────────────────────────────
        Complex[] productComplex = ifft(productFFT);

        // ─────────────────────────────────────────────────────────────
        // STEP 6: Extract real parts (imaginary should be ~0)
        // Due to numerical errors, imaginary parts may be tiny non-zero
        // ─────────────────────────────────────────────────────────────
        double[] result = new double[resultDegree];

        for (int i = 0; i < resultDegree; i++) {
            result[i] = productComplex[i].real();

            // Clean up numerical noise: very small values → 0
            // This restores the mathematical property that
            // product of real polynomials has real coefficients
            if (Math.abs(result[i]) < 1e-10) {
                result[i] = 0.0;
            }
        }

        return result;
    }

    /**
     * Find smallest power of 2 greater than or equal to n.
     *
     * Examples:
     *   nextPowerOf2(5) = 8
     *   nextPowerOf2(8) = 8
     *   nextPowerOf2(9) = 16
     *
     * @param n input value
     * @return smallest 2^k where 2^k ≥ n
     */
    public static int nextPowerOf2(int n) {
        // Handle edge case
        if (n <= 0) return 1;

        // Double until we reach or exceed n
        int power = 1;
        while (power < n) {
            power *= 2;  // power = power << 1
        }

        return power;
    }
}