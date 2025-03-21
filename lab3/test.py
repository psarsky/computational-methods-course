import mpmath
from bisection_method import bisection_method
from test_functions import f_1, f_2, f_3


precision = int(input("Enter precision: "))
tol = float(input("Enter tolerance: "))
iter_limit = int(input("Enter iteration limit: "))

print("f_1: cos(x)cosh(x) - 1")
a = 1.5 * mpmath.pi
b = 2 * mpmath.pi
root, iters = bisection_method(f_1, a, b, precision, tol, iter_limit)
print(f"Root: {root}")
print(f"Iterations: {iters}")

print("f_2: 1/x - tan(x)")
a = 1e-10
b = 0.5 * mpmath.pi
root, iters = bisection_method(f_2, a, b, precision, tol, iter_limit)
print(f"Root: {root}")
print(f"Iterations: {iters}")

print("f_3: 2^(-x) + exp(x) + 2cos(x) - 6")
a = 1
b = 3
root, iters = bisection_method(f_3, a, b, precision, tol, iter_limit)
print(f"Root: {root}")
print(f"Iterations: {iters}")