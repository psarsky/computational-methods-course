"""LU factorization using Gauss-Jordan upper triangular function."""
import time

import numpy as np


def lu_factorization(matrix):
    """Performs LU factorization using Gauss-Jordan upper triangular function."""
    matrix_size = len(matrix)
    lower = np.eye(matrix_size)
    upper = matrix.copy()

    for i in range(matrix_size):
        for j in range(i + 1, matrix_size):
            lower[j, i] = upper[j, i] / upper[i, i]
            upper[j] -= upper[i] * lower[j, i]

    return lower, upper


def main():
    """Tests LU factorization correctness by computing ||matrix - LU||."""
    sizes = [500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 2137]
    low, high = -10**10, 10**10

    for matrix_size in sizes:
        matrix = np.random.uniform(low, high, size=(matrix_size, matrix_size))
        matrix_orig = matrix.copy()

        start = time.time()
        lower, upper = lu_factorization(matrix)
        end = time.time()

        error = np.linalg.norm(matrix_orig - lower @ upper)
        print(f"Size {matrix_size}x{matrix_size}: Absolute error = {error:.6e}, "
              f"Relative error = {(error*100)/high:.10f}%, Time = {end - start:.4f}s")


if __name__ == "__main__":
    main()
