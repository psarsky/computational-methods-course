import time

import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import eig


def power_method(A, max_iter=10000, eps=1e-15):
    n = A.shape[0]

    x = np.random.rand(n)
    x = x / np.max(np.abs(x))
    x_new = np.zeros(n)

    eigenvalue = 0

    for i in range(max_iter):
        x_new = A @ x

        max_abs = np.max(np.abs(x_new))
        x_new = x_new / max_abs

        if np.max(np.abs(x_new - x)) < eps:
            break

        x = x_new

    eigenvalue = max_abs
    eigenvector = x_new / np.linalg.norm(x_new)

    return eigenvalue, eigenvector, i + 1


def compare_with_library(A):
    start_time = time.time()
    eigen_val, eigen_vec, iterations = power_method(A)
    custom_time = time.time() - start_time

    start_time = time.time()
    lib_vals, lib_vecs = eig(A)
    lib_time = time.time() - start_time

    idx = np.argmax(np.abs(lib_vals))
    lib_val = lib_vals[idx]
    lib_vec = lib_vecs[:, idx]

    if np.dot(eigen_vec, lib_vec) < 0:
        lib_vec = -lib_vec

    rel_error_val = abs(eigen_val - lib_val) / abs(lib_val) if lib_val != 0 else abs(eigen_val)

    vec_similarity = abs(np.dot(eigen_vec, lib_vec))

    return {
        'custom_eigenvalue': eigen_val,
        'library_eigenvalue': lib_val,
        'relative_error_val': rel_error_val,
        'vector_similarity': vec_similarity,
        'iterations': iterations,
        'custom_time': custom_time,
        'library_time': lib_time
    }


def benchmark_sizes():
    sizes = [100, 200, 500, 1000, 2000]
    custom_times = []
    library_times = []

    for size in sizes:
        A = np.random.rand(size, size)
        A = A + A.T

        result = compare_with_library(A)
        custom_times.append(result['custom_time'])
        library_times.append(result['library_time'])

        print(f"Matrix size: {size}x{size}")
        print(f"Eigenvalue (custom):  {result['custom_eigenvalue']:.12f}")
        print(f"Eigenvalue (library): {result['library_eigenvalue']:.12f}")
        print(f"Relative eigenvalue error: {result['relative_error_val']:.6e}")
        print(f"Eigenvector similarity: {result['vector_similarity']:.12f}")
        print(f"Number of iterations: {result['iterations']}")
        print(f"Custom implementation time:  {result['custom_time']:.6f}s")
        print(f"Library implementation time: {result['library_time']:.6f}s")
        print("-" * 50)

    return sizes, custom_times, library_times


def plot_benchmark(sizes, custom_times, library_times):
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, custom_times, 'o-', label='Custom implementation')
    plt.plot(sizes, library_times, 's-', label='Library implementation')
    plt.xlabel('Matrix size (n x n)')
    plt.ylabel('Execution time [s]')
    plt.title('Power Method Execution Time Comparison')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    print("\nBenchmark for different matrix sizes:")
    sizes_, custom_times_, library_times_ = benchmark_sizes()

    plot_benchmark(sizes_, custom_times_, library_times_)
