"""Test the root-finding methods on the test functions."""
import time

import matplotlib.pyplot as plt
import mpmath
from methods import (bisection, bisection_vis, combo, combo_vis, newton,
                     newton_vis, secant, secant_vis)
from test_functions import df_1, df_2, df_3, f_1, f_2, f_3


def print_bisection(f, a, b, precision, tol, iter_limit):
    """Print the results of the bisection method."""
    print("")
    print("Bisection method:")
    start = time.time()
    root, iters = bisection(f, a, b, precision, tol, iter_limit)
    t = time.time() - start
    print(f"Root: {root}")
    print(f"Iterations: {iters}")
    print(f"Time: {t:.5f}s")


def print_newton(f, df, b, precision, tol, iter_limit):
    """Print the results of Newton's method."""
    print("")
    print("Newton method:")
    start = time.time()
    root, iters = newton(f, df, b, precision, tol, iter_limit)
    t = time.time() - start
    print(f"Root: {root}")
    print(f"Iterations: {iters}")
    print(f"Time: {t:.5f}s")


def print_secant(f, a, b, precision, tol, iter_limit):
    """Print the results of the secant method."""
    print("")
    print("Secant method:")
    start = time.time()
    root, iters = secant(f, a, b, precision, tol, iter_limit)
    t = time.time() - start
    print(f"Root: {root}")
    print(f"Iterations: {iters}")
    print(f"Time: {t:.5f}s")


def print_combo(f, df, a, b, precision, tol, iter_limit):
    """Print the results of the bisection + Newton method."""
    print("")
    print("Bisection + Newton method:")
    start = time.time()
    root, iters = combo(f, df, a, b, precision, tol, iter_limit)
    t = time.time() - start
    print(f"Root: {root}")
    print(f"Iterations: {iters}")
    print(f"Time: {t:.5f}s")


def main():
    """Test the root-finding methods on the test functions."""
    precision = int(input("Enter precision: "))
    tol = float(input("Enter tolerance: "))
    iter_limit = int(input("Enter iteration limit: "))

    print("\n")

    print("f_1: cos(x)cosh(x) - 1")
    a = 1.5 * mpmath.pi
    b = 2 * mpmath.pi
    print_bisection(f_1, a, b, precision, tol, iter_limit)
    print_newton(f_1, df_1, b, precision, tol, iter_limit)
    print_secant(f_1, a, b, precision, tol, iter_limit)
    print_combo(f_1, df_1, a, b, precision, tol, iter_limit)

    bisection_steps = bisection_vis(f_1, a, b, precision, tol, iter_limit)
    newton_steps = newton_vis(f_1, df_1, b, precision, tol, iter_limit)
    secant_steps = secant_vis(f_1, a, b, precision, tol, iter_limit)
    combo_steps = combo_vis(f_1, df_1, a, b, precision, tol, iter_limit)

    plt.figure(figsize=(8, 6))
    plt.plot(bisection_steps, label="Bisection method", marker='o')
    plt.plot(newton_steps, label="Newton method", marker='s')
    plt.plot(secant_steps, label="Secant method", marker='^')
    plt.plot(combo_steps, label="Bisection + Newton method", marker='v')
    plt.xlabel("Iteration")
    plt.ylabel("Root Approximation")
    plt.title("Convergence of Root-Finding Methods")
    plt.legend()
    plt.grid()

    print("\n====================================\n")

    print("f_2: 1/x - tan(x)")
    a = 1e-10
    b = 0.5 * mpmath.pi

    print_bisection(f_2, a, b, precision, tol, iter_limit)
    print_newton(f_2, df_2, b / 2, precision, tol, iter_limit)
    print_secant(f_2, a, b, precision, tol, iter_limit)
    print_combo(f_2, df_2, a, b, precision, tol, iter_limit)

    bisection_steps = bisection_vis(f_2, a, b, precision, tol, iter_limit)
    newton_steps = newton_vis(f_2, df_2, b, precision, tol, iter_limit)
    secant_steps = secant_vis(f_2, a, b, precision, tol, iter_limit)
    combo_steps = combo_vis(f_2, df_2, a, b, precision, tol, iter_limit)

    plt.figure(figsize=(8, 6))
    plt.plot(bisection_steps, label="Bisection method", marker='o')
    plt.plot(newton_steps, label="Newton method", marker='s')
    plt.plot(secant_steps, label="Secant method", marker='^')
    plt.plot(combo_steps, label="Bisection + Newton method", marker='v')
    plt.xlabel("Iteration")
    plt.ylabel("Root Approximation")
    plt.title("Convergence of Root-Finding Methods")
    plt.legend()
    plt.grid()

    print("\n====================================\n")

    print("f_3: 2^(-x) + exp(x) + 2cos(x) - 6")
    a = 1
    b = 3

    print_bisection(f_3, a, b, precision, tol, iter_limit)
    print_newton(f_3, df_3, b, precision, tol, iter_limit)
    print_secant(f_3, a, b, precision, tol, iter_limit)
    print_combo(f_3, df_3, a, b, precision, tol, iter_limit)

    bisection_steps = bisection_vis(f_3, a, b, precision, tol, iter_limit)
    newton_steps = newton_vis(f_3, df_3, b, precision, tol, iter_limit)
    secant_steps = secant_vis(f_3, a, b, precision, tol, iter_limit)
    combo_steps = combo_vis(f_3, df_3, a, b, precision, tol, iter_limit)

    plt.figure(figsize=(8, 6))
    plt.plot(bisection_steps, label="Bisection method", marker='o')
    plt.plot(newton_steps, label="Newton method", marker='s')
    plt.plot(secant_steps, label="Secant method", marker='^')
    plt.plot(combo_steps, label="Bisection + Newton method", marker='v')
    plt.xlabel("Iteration")
    plt.ylabel("Root Approximation")
    plt.title("Convergence of Root-Finding Methods")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()
