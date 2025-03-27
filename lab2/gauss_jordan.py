"""Gauss-Jordan elimination method for solving linear systems of equations."""
import time

import numpy as np


def gauss_jordan_upper(matrix, vector):
    """Transforms input matrix into an upper triangular matrix using Gauss-Jordan elimination."""
    matrix_size = len(matrix)
    matrix_extended = np.hstack([matrix.astype(float), vector.reshape(-1, 1).astype(float)])

    for i in range(matrix_size):
        max_row = np.argmax(abs(matrix_extended[i:, i])) + i
        if i != max_row:
            matrix_extended[[i, max_row]] = matrix_extended[[max_row, i]]

        matrix_extended[i] /= matrix_extended[i, i]

        for j in range(i + 1, matrix_size):
            matrix_extended[j] -= matrix_extended[i] * matrix_extended[j, i]

    return matrix_extended[:, :-1], matrix_extended[:, -1]


def back_substitution(matrix_upper, vector):
    """Solves Ux = y for an upper triangular matrix using back substitution."""
    matrix_size = len(matrix_upper)
    result = np.zeros(matrix_size)

    for i in range(matrix_size - 1, -1, -1):
        result[i] = (vector[i] - np.dot(matrix_upper[i, i+1:], result[i+1:])) / matrix_upper[i, i]

    return result


def solve_system(matrix, vector):
    """Solves Ax = y using Gauss-Jordan elimination followed by back substitution."""
    matrix_upper, vector_new = gauss_jordan_upper(matrix, vector)
    return back_substitution(matrix_upper, vector_new)


def main():
    """Main function - compares Gauss-Jordan elimination method with NumPy's solver."""
    sizes = [500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 2137]

    for matrix_size in sizes:
        matrix = np.random.rand(matrix_size, matrix_size)
        vector = np.random.rand(matrix_size)

        start = time.time()
        gj_result = solve_system(matrix, vector)
        gj_time = time.time() - start

        start = time.time()
        np_result = np.linalg.solve(matrix, vector)
        np_time = time.time() - start

        success = np.allclose(gj_result, np_result)

        print(f"Size {matrix_size}x{matrix_size}: Gauss-Jordan = {gj_time:.4f}s, NumPy = {np_time:.4f}s, "
              f"Ratio = {gj_time / np_time:.4f}, Success = {success}")


if __name__ == "__main__":
    main()
