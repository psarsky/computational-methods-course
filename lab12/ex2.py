import math

import numpy as np
from scipy import integrate
from test_functions import f1, f2, f3, f4, f5


def simpson_1d(f, a, b, n):
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


def simpson_2d(f, x_range, y_range, nx, ny):
    if nx % 2 != 0:
        nx += 1
    if ny % 2 != 0:
        ny += 1

    ax, bx = x_range
    ay, by = y_range

    hx = (bx - ax) / nx
    hy = (by - ay) / ny

    x = np.linspace(ax, bx, nx + 1)
    y = np.linspace(ay, by, ny + 1)

    result = 0

    for i in range(nx + 1):
        for j in range(ny + 1):
            coeff = 1

            if i in (0, nx):
                coeff *= 1
            elif i % 2 == 1:
                coeff *= 4
            else:
                coeff *= 2

            if j in (0, ny):
                coeff *= 1
            elif j % 2 == 1:
                coeff *= 4
            else:
                coeff *= 2

            result += coeff * f(x[i], y[j])

    return result * hx * hy / 9


def main():
    test_cases = [
        (f1, 0.1, 2.0, "∫exp(-x²)·(ln(x))² dx"),
        (f2, 3.0, 5.0, "∫1/(x³-2x-5) dx"),
        (f3, 0.0, math.pi, "∫x⁵·exp(-x)·sin(x) dx"),
    ]

    for func, a, b, description in test_cases:
        print(f"\nFunkcja: {description} od {a} do {b}")
        print("-" * 50)

        quad_result, quad_error = integrate.quad(func, a, b)
        print(f"scipy.quad:      {quad_result:.10f} (błąd est: {quad_error:.2e})")

        n_values = [10, 50, 100, 200, 500]

        for n in n_values:
            simpson_result = simpson_1d(func, a, b, n)
            error = abs(simpson_result - quad_result)
            relative_error = error / abs(quad_result) * 100 if quad_result != 0 else 0

            print(
                f"Simpson (n={n:3d}): {simpson_result:.10f} "
                f"(błąd: {error:.2e}, {relative_error:.4f}%)"
            )
    
    test_cases_2d = [
        (f4, (0.1, 0.9), (0.1, 0.8), "∫∫1/(√(x+y)·(1+x+y)) dxdy"),
        (f5, (-3, 3), (-5, 5), "∫∫(x² + y²) dxdy"),
    ]

    for func, x_range, y_range, description in test_cases_2d:
        print(f"\nFunkcja: {description}")
        print(f"Przedział: x∈[{x_range[0]}, {x_range[1]}], y∈[{y_range[0]}, {y_range[1]}]")
        print("-" * 50)

        quad_2d_result, quad_2d_error = integrate.dblquad(func, y_range[0], y_range[1], x_range[0], x_range[1])
        print(f"scipy.dblquad:   {quad_2d_result:.10f} (błąd est: {quad_2d_error:.2e})")

        n_values_2d = [(10, 10), (50, 50), (100, 100), (200, 200), (500, 500)]

        for nx, ny in n_values_2d:
            simpson_2d_result = simpson_2d(func, x_range, y_range, nx, ny)
            error_2d = abs(simpson_2d_result - quad_2d_result)
            relative_error_2d = (
                error_2d / abs(quad_2d_result) * 100 if quad_2d_result != 0 else 0
            )

            print(
                f"Simpson ({nx}×{ny}):   {simpson_2d_result:.10f} "
                f"(błąd: {error_2d:.2e}, {relative_error_2d:.4f}%)"
            )


if __name__ == "__main__":
    main()
