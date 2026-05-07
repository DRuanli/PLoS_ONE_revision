package infrastructure.math;

/**
 * Complex - Immutable complex number representation for FFT computations.
 *
 * A complex number has the form: a + bi where:
 *   - a is the real part
 *   - b is the imaginary part
 *   - i is the imaginary unit (i² = -1)
 *
 * Why needed for FFT?
 *   FFT operates in the complex domain using roots of unity:
 *   e^(iθ) = cos(θ) + i·sin(θ)  (Euler's formula)
 *
 * Immutability:
 *   All operations return NEW Complex objects.
 *   Original objects are never modified.
 *   This ensures thread safety and predictable behavior.
 *
 * @author Dang Nguyen Le, Gia Huy Vo
 */
public class Complex {

    /**
     * Real part of complex number (the 'a' in a + bi).
     */
    private final double real;

    /**
     * Imaginary part of complex number (the 'b' in a + bi).
     */
    private final double imag;

    /**
     * Constructor for complex number a + bi.
     *
     * @param real the real part (a)
     * @param imag the imaginary part (b)
     */
    public Complex(double real, double imag) {
        this.real = real;
        this.imag = imag;
    }

    /**
     * Constructor for real number (imaginary part = 0).
     * Convenience constructor for converting real to complex.
     *
     * @param real the real value
     */
    public Complex(double real) {
        // Real number is complex with imag = 0
        this(real, 0.0);
    }

    /**
     * Get real part of this complex number.
     *
     * @return real part
     */
    public double real() {
        return real;
    }

    /**
     * Complex addition: (a + bi) + (c + di) = (a+c) + (b+d)i
     *
     * @param that complex number to add
     * @return new Complex representing this + that
     */
    public Complex plus(Complex that) {
        // Add real parts and imaginary parts separately
        return new Complex(this.real + that.real, this.imag + that.imag);
    }

    /**
     * Complex subtraction: (a + bi) - (c + di) = (a-c) + (b-d)i
     *
     * @param that complex number to subtract
     * @return new Complex representing this - that
     */
    public Complex minus(Complex that) {
        // Subtract real parts and imaginary parts separately
        return new Complex(this.real - that.real, this.imag - that.imag);
    }

    /**
     * Complex multiplication: (a + bi) × (c + di) = (ac - bd) + (ad + bc)i
     *
     * Derivation using FOIL:
     *   (a + bi)(c + di) = ac + adi + bci + bdi²
     *                    = ac + adi + bci - bd    (since i² = -1)
     *                    = (ac - bd) + (ad + bc)i
     *
     * @param that complex number to multiply
     * @return new Complex representing this × that
     */
    public Complex times(Complex that) {
        // Real part: ac - bd
        double newReal = this.real * that.real - this.imag * that.imag;

        // Imaginary part: ad + bc
        double newImag = this.real * that.imag + this.imag * that.real;

        return new Complex(newReal, newImag);
    }

    /**
     * Scalar multiplication: (a + bi) × k = ka + kbi
     *
     * @param scalar real number to multiply by
     * @return new Complex representing this × scalar
     */
    public Complex times(double scalar) {
        return new Complex(this.real * scalar, this.imag * scalar);
    }

    /**
     * Complex conjugate: conjugate(a + bi) = a - bi
     *
     * Property: z × conjugate(z) = |z|² (always real and non-negative)
     *
     * Used in inverse FFT:
     *   IFFT(x) = conjugate(FFT(conjugate(x))) / n
     *
     * @return new Complex representing conjugate of this
     */
    public Complex conjugate() {
        // Negate imaginary part
        return new Complex(real, -imag);
    }

    /**
     * String representation for debugging.
     *
     * @return string like "3.0 + 4.0i" or "5.0" or "2.0i"
     */
    @Override
    public String toString() {
        // Pure real number
        if (imag == 0) return real + "";

        // Pure imaginary number
        if (real == 0) return imag + "i";

        // Negative imaginary part: a - |b|i
        if (imag < 0) return real + " - " + (-imag) + "i";

        // Positive imaginary part: a + bi
        return real + " + " + imag + "i";
    }

    /**
     * Equality check for complex numbers.
     *
     * @param other object to compare
     * @return true if both real and imaginary parts are equal
     */
    @Override
    public boolean equals(Object other) {
        // Null check
        if (other == null) return false;

        // Type check
        if (other.getClass() != this.getClass()) return false;

        Complex that = (Complex) other;

        // Compare both parts
        return (this.real == that.real) && (this.imag == that.imag);
    }

    /**
     * Hash code for use in HashMap/HashSet.
     *
     * @return hash code based on real and imaginary parts
     */
    @Override
    public int hashCode() {
        // Combine hashes of both parts
        return Double.hashCode(real) * 31 + Double.hashCode(imag);
    }
}