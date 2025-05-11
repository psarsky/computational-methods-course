"""Eigendecomposition of a matrix"""

import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.linalg import eig
from scipy.linalg import lu_factor, lu_solve

MAX_ITER = 1000
EPS = 1e-10


def power_method(A, max_iter=MAX_ITER, tol=EPS):
    """Power method for finding the largest eigenvalue and corresponding eigenvector of a matrix A."""
    n = A.shape[0]

    x = np.random.rand(n)
    x = x / np.max(np.abs(x))
    x_new = np.zeros(n)

    eigenvalue = 0

    for i in range(max_iter):
        x_new = A @ x

        max_abs = np.max(np.abs(x_new))
        x_new = x_new / max_abs

        if np.max(np.abs(x_new - x)) < tol:
            break

        x = x_new

    eigenvalue = max_abs
    eigenvector = x_new / np.linalg.norm(x_new)

    return eigenvalue, eigenvector, i + 1


def inverse_power_method(A, sigma, max_iter=MAX_ITER, tol=EPS):
    """Inverse power method which utilizes LU factorization."""
    n = A.shape[0]

    A_shifted = A - sigma * np.eye(n)

    lu, piv = lu_factor(A_shifted)

    x = np.random.rand(n)
    x = x / np.max(np.abs(x))

    eigenvalue = 0
    for i in range(max_iter):
        x_new = lu_solve((lu, piv), x)

        max_abs = np.max(np.abs(x_new))
        x_new = x_new / max_abs

        mu = 1.0 / max_abs
        eigenvalue = sigma + mu

        if np.max(np.abs(x_new - x)) < tol:
            break

        x = x_new

    eigenvector = x_new / np.linalg.norm(x_new)

    return eigenvalue, eigenvector, i + 1


def rayleigh_quotient_iteration(A, max_iter=MAX_ITER, tol=EPS):
    """Rayleigh quotient iteration method for finding eigenvalue and eigenvector."""
    n = A.shape[0]

    x = np.random.rand(n)
    x = x / np.linalg.norm(x)

    mu = (x.T @ A @ x) / (x.T @ x)

    for i in range(max_iter):
        A_shifted = A - mu * np.eye(n)

        lu, piv = lu_factor(A_shifted)
        y = lu_solve((lu, piv), x)

        x_new = y / np.linalg.norm(y)

        mu_new = (x_new.T @ A @ x_new) / (x_new.T @ x_new)

        if np.max(np.abs(x_new - x)) < tol:
            break

        x = x_new
        mu = mu_new

    return mu, x, i + 1


def compare_methods(A):
    """Compares all methods with library function."""
    results = {}

    start_time = time.time()
    lib_vals, lib_vecs = eig(A)
    lib_time = time.time() - start_time

    idx = np.argmax(np.abs(lib_vals))
    lib_val = lib_vals[idx]
    lib_vec = lib_vecs[:, idx]

    results["library"] = {
        "eigenvalue": lib_val,
        "eigenvector": lib_vec,
        "time": lib_time,
    }

    start_time = time.time()
    pm_val, pm_vec, pm_iter = power_method(A)
    pm_time = time.time() - start_time

    if np.dot(pm_vec, lib_vec) < 0:
        pm_vec = -pm_vec

    results["power_method"] = {
        "eigenvalue": pm_val,
        "eigenvector": pm_vec,
        "iterations": pm_iter,
        "time": pm_time,
        "rel_error_val": (
            abs(pm_val - lib_val) / abs(lib_val) if lib_val != 0 else abs(pm_val)
        ),
        "vec_similarity": abs(np.dot(pm_vec, lib_vec)),
    }

    sigma = lib_val * 0.95
    start_time = time.time()
    ipm_val, ipm_vec, ipm_iter = inverse_power_method(A, sigma)
    ipm_time = time.time() - start_time

    if np.dot(ipm_vec, lib_vec) < 0:
        ipm_vec = -ipm_vec

    results["inverse_power_method"] = {
        "eigenvalue": ipm_val,
        "eigenvector": ipm_vec,
        "iterations": ipm_iter,
        "time": ipm_time,
        "rel_error_val": (
            abs(ipm_val - lib_val) / abs(lib_val) if lib_val != 0 else abs(ipm_val)
        ),
        "vec_similarity": abs(np.dot(ipm_vec, lib_vec)),
    }

    start_time = time.time()
    rqi_val, rqi_vec, rqi_iter = rayleigh_quotient_iteration(A)
    rqi_time = time.time() - start_time

    if np.dot(rqi_vec, lib_vec) < 0:
        rqi_vec = -rqi_vec

    results["rayleigh_quotient"] = {
        "eigenvalue": rqi_val,
        "eigenvector": rqi_vec,
        "iterations": rqi_iter,
        "time": rqi_time,
        "rel_error_val": (
            abs(rqi_val - lib_val) / abs(lib_val) if lib_val != 0 else abs(rqi_val)
        ),
        "vec_similarity": abs(np.dot(rqi_vec, lib_vec)),
    }

    return results


def benchmark_sizes():
    """Benchmarks all implementations against the library function for different matrix sizes."""
    sizes = [100, 200, 500, 1000]
    methods = ["power_method", "inverse_power_method", "rayleigh_quotient", "library"]

    times = {method: [] for method in methods}
    errors = {method: [] for method in methods if method != "library"}
    iterations = {method: [] for method in methods if method != "library"}
    all_results = []

    for size in sizes:
        print(f"\nMatrix size: {size}x{size}")
        print("-" * 50)

        A = np.random.rand(size, size)
        A = (A + A.T) / 2

        results = compare_methods(A)

        size_results = []
        for method in methods:
            times[method].append(results[method]["time"])

            if method != "library":
                errors[method].append(results[method]["rel_error_val"])
                iterations[method].append(results[method]["iterations"])

                row = {
                    "Matrix Size": f"{size}x{size}",
                    "Method": method.replace("_", " ").title(),
                    "Eigenvalue": f"{results[method]['eigenvalue']:.12f}",
                    "Relative Error": f"{results[method]['rel_error_val']:.6e}",
                    "Vector Similarity": f"{results[method]['vec_similarity']:.12f}",
                    "Iterations": results[method]["iterations"],
                    "Time (s)": f"{results[method]['time']:.6f}",
                }
                size_results.append(row)
            else:
                row = {
                    "Matrix Size": f"{size}x{size}",
                    "Method": "Library (NumPy)",
                    "Eigenvalue": f"{results[method]['eigenvalue']:.12f}",
                    "Relative Error": "N/A",
                    "Vector Similarity": "N/A",
                    "Iterations": "N/A",
                    "Time (s)": f"{results[method]['time']:.6f}",
                }
                size_results.append(row)

        df = pd.DataFrame(size_results)
        print(df.to_string(index=False))
        all_results.extend(size_results)

    summary_df = pd.DataFrame(all_results)
    print("\nSummary of all results:")
    print(summary_df.to_string(index=False))

    return sizes, times, errors, iterations


def plot_benchmark(sizes, times, errors, iterations):
    """Plots the benchmark results."""
    methods = ["power_method", "inverse_power_method", "rayleigh_quotient", "library"]
    method_labels = [
        "Power Method",
        "Inverse Power Method",
        "Rayleigh Quotient",
        "Library (NumPy)",
    ]

    plt.figure(figsize=(12, 6))
    for i, method in enumerate(methods):
        plt.plot(
            sizes, times[method], marker=["o", "s", "^", "d"][i], label=method_labels[i]
        )

    plt.xlabel("Matrix size (n x n)")
    plt.ylabel("Execution time [s]")
    plt.title("Eigendecomposition Methods - Execution Time Comparison")
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
    plt.title("Eigendecomposition Methods - Relative Error Comparison")
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
    plt.title("Eigendecomposition Methods - Iteration Count Comparison")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("\nBenchmarking eigendecomposition methods for different matrix sizes:")
    sizes_, times_, errors_, iterations_ = benchmark_sizes()

    plot_benchmark(sizes_, times_, errors_, iterations_)
