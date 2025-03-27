"""Exercise 4b: Single vs double precision comparison for logistic map"""
import numpy as np
import matplotlib.pyplot as plt


def logistic_map_single(x, r):
    """Compute the logistic map in single precision."""
    return np.float32(r * x * (1 - x))


def logistic_map_double(x, r):
    """Compute the logistic map in double precision."""
    return r * x * (1 - x)


r_values = [3.75, 3.76, 3.77, 3.78, 3.79, 3.8]
x0 = 0.4
iterations = 100

for r_val in r_values:
    plt.figure(figsize=(12, 6))

    x_single = np.float32(x0)
    x_double = float(x0)

    x_vals_single = []
    x_vals_double = []

    for _ in range(iterations):
        x_single = logistic_map_single(x_single, np.float32(r_val))
        x_double = logistic_map_double(x_double, r_val)
        x_vals_single.append(x_single)
        x_vals_double.append(x_double)

    plt.scatter(range(iterations), x_vals_single, label=f'r={r_val:.2f} (single)', marker='.', color="orange")
    plt.plot(range(iterations), x_vals_double, label=f'r={r_val:.2f} (double)', marker='.')

    for i in range(iterations):
        color = 'g' if x_vals_single[i] > x_vals_double[i] else 'r'
        plt.plot([i, i], [x_vals_single[i], x_vals_double[i]], color, linewidth=0.5)

    plt.xlabel("Iterations")
    plt.ylabel("$x_n$")
    plt.title("Single and double precision trajectory comparison")
    plt.legend()

plt.show()

# Initially, both precisions produce similar results, but differences become evident after multiple iterations.
# Single precision accumulates rounding errors faster than double precision, leading to trajectory divergence over
# iterations.
