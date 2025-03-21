import mpmath
import time
from methods import bisection, newton
from test_functions import f_1, f_2, f_3, df_1, df_2, df_3


def print_bisection(f, a, b, precision, tol, iter_limit):
    print("")
    print("Bisection method:")
    start = time.time()
    root, iters = bisection(f, a, b, precision, tol, iter_limit)
    t = time.time() - start
    print(f"Root: {root}")
    print(f"Iterations: {iters}")
    print(f"Time: {t:.5f}s")


def print_newton(f, df, b, precision, tol, iter_limit):
    print("")
    print("Newton method:")
    start = time.time()
    root, iters = newton(f, df, b, precision, tol, iter_limit)
    t = time.time() - start
    print(f"Root: {root}")
    print(f"Iterations: {iters}")
    print(f"Time: {t:.5f}s")


def main():
    precision = int(input("Enter precision: "))
    tol = float(input("Enter tolerance: "))
    iter_limit = int(input("Enter iteration limit: "))

    print("\n")

    print("f_1: cos(x)cosh(x) - 1")
    a = 1.5 * mpmath.pi
    b = 2 * mpmath.pi
    print_bisection(f_1, a, b, precision, tol, iter_limit)
    print_newton(f_1, df_1, b, precision, tol, iter_limit)

    print("\n====================================\n")

    print("f_2: 1/x - tan(x)")
    a = 1e-10
    b = 0.5 * mpmath.pi

    print_bisection(f_2, a, b, precision, tol, iter_limit)
    print_newton(f_2, df_2, b / 2, precision, tol, iter_limit)

    print("\n====================================\n")

    print("f_3: 2^(-x) + exp(x) + 2cos(x) - 6")
    a = 1
    b = 3

    print_bisection(f_3, a, b, precision, tol, iter_limit)
    print_newton(f_3, df_3, b, precision, tol, iter_limit)

if __name__ == "__main__":
    main()
