"""Numerical integration using Simpson's rule and comparison with scipy's quad."""

import math

import numpy as np
from scipy import integrate
from test_functions import f1, f2, f3


def simpson(f, a, b, n):
    """Simpson's 1/3 rule for numerical integration."""
    if n % 2 != 0:
        n += 1

    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = f(x)

    result = y[0] + y[-1]

    for i in range(1, n, 2):
        result += 4 * y[i]

    for i in range(2, n, 2):
        result += 2 * y[i]

    return result * h / 3


def main():
    """Main function to test numerical integration methods."""
    test_cases = [
        (f1, 0.1, 2.0, "∫exp(-x²)·(ln(x))² dx"),
        (f2, 3.0, 5.0, "∫1/(x³-2x-5) dx"),
        (f3, 0.0, math.pi, "∫x⁵·exp(-x)·sin(x) dx"),
    ]

    for func, a, b, description in test_cases:
        print(f"\nIntegral: {description} from {a} to {b}")
        print("-" * 50)

        quad_result, quad_error = integrate.quad(func, a, b)
        print(f"scipy.quad:      {quad_result:.10f} (error: {quad_error:.2e})")

        n_values = [10, 50, 100, 200, 500]

        for n in n_values:
            simpson_result = simpson(func, a, b, n)
            error = abs(simpson_result - quad_result)
            relative_error = error / abs(quad_result) * 100 if quad_result != 0 else 0

            print(
                f"Simpson (n={n:3d}): {simpson_result:.10f} "
                f"(error: {error:.2e}, {relative_error:.4f}%)"
            )


if __name__ == "__main__":
    main()
