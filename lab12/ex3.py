"""Numerical calculation of double integrals."""

import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate
from test_functions import f4, f5


def task_1():
    """Compare iterated and tiled double integrals."""
    result_iter, error_iter = integrate.dblquad(f4, 0, 1, 0, lambda x: 1 - x)
    result_tiled, error_tiled = integrate.dblquad(f4, 0, 1, 0, lambda x: 1 - x)

    print("Integral: ∫∫1/(√(x+y)·(1+x+y)) dxdy")
    print(f"Iterated: {result_iter:.10f}, error: {error_iter:.2e}")
    print(f"Tiled: {result_tiled:.10f}, error: {error_tiled:.2e}")
    print(f"Difference between methods: {abs(result_iter - result_tiled):.2e}")
    print()


def task_2():
    """Numerical integration using double trapezoidal rule."""
    x = np.linspace(-3, 3, 20)
    y = np.linspace(-5, 5, 20)
    X, Y = np.meshgrid(x, y)
    Z = f5(X, Y)

    results_trapz = []
    grid_sizes = [5, 10, 20, 50, 100]

    for n in grid_sizes:
        x_grid = np.linspace(-3, 3, n)
        y_grid = np.linspace(-5, 5, n)
        X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
        Z_grid = f5(X_grid, Y_grid)

        result = np.trapezoid(np.trapezoid(Z_grid, y_grid, axis=0), x_grid)
        results_trapz.append(result)

    result_integral2, error_integral2 = integrate.dblquad(f5, -3, 3, -5, 5)

    print("Integral: ∫∫(x² + y²) dxdy (trapezoidal rule)")
    for _, (n, result) in enumerate(zip(grid_sizes, results_trapz)):
        print(f"{n}x{n} mesh: {result:.10f}")

    print(f"\nReference value (integral2): {result_integral2:.10f}")
    print(f"integral2 method error: {error_integral2:.2e}")

    print("\nAccuracy comparison:")
    for _, (n, result) in enumerate(zip(grid_sizes, results_trapz)):
        error = abs(result - result_integral2)
        print(f"{n}x{n} mesh - error: {error:.2e}")

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(grid_sizes, [abs(r - result_integral2) for r in results_trapz], "bo-")
    plt.xlabel("Mesh size (n)")
    plt.ylabel("Error")
    plt.title("Trapezodial rule error vs mesh size")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.contourf(X, Y, Z, levels=20, cmap="viridis")
    plt.colorbar(label="f5(x,y) = x² + y²")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Function visualization")
    plt.xlim(-3, 3)
    plt.ylim(-5, 5)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    task_1()
    task_2()
