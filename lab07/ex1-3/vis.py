"""Visualization function for the eigendecomposition benchmark."""

import matplotlib.pyplot as plt


def plot_benchmark(sizes, times, errors, iterations):
    """Plots the benchmark results."""
    methods = ["power_method", "inverse_power_method", "rayleigh_quotient", "library"]
    method_labels = [
        "Power method",
        "Inverse power method",
        "Rayleigh quotient",
        "Library (NumPy)",
    ]

    plt.figure(figsize=(12, 6))
    for i, method in enumerate(methods):
        plt.plot(
            sizes, times[method], marker=["o", "s", "^", "d"][i], label=method_labels[i]
        )

    plt.xlabel("Matrix size (n x n)")
    plt.ylabel("Execution time [s]")
    plt.title("Eigendecomposition methods - execution time comparison")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 6))
    for i, method in enumerate(methods[:-1]):
        plt.semilogy(
            sizes, errors[method], marker=["o", "s", "^"][i], label=method_labels[i]
        )

    plt.xlabel("Matrix size (n x n)")
    plt.ylabel("Relative error")
    plt.title("Eigendecomposition methods - relative error comparison")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 6))
    for i, method in enumerate(methods[:-1]):
        plt.plot(
            sizes, iterations[method], marker=["o", "s", "^"][i], label=method_labels[i]
        )

    plt.xlabel("Matrix size (n x n)")
    plt.ylabel("Number of iterations")
    plt.title("Eigendecomposition methods - iteration count comparison")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
