"""QR factorization and stability analysis"""

import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import cond, norm, qr


def gram_schmidt_qr(A):
    """Perform QR factorization using the Gram-Schmidt method."""
    Q = np.zeros(A.shape)
    R = np.zeros((A.shape[1], A.shape[1]))

    for k in range(A.shape[1]):
        u_k = A[:, k].copy()

        for i in range(k):
            R[i, k] = np.dot(Q[:, i], A[:, k])
            u_k = u_k - R[i, k] * Q[:, i]

        R[k, k] = np.linalg.norm(u_k)
        Q[:, k] = u_k / R[k, k]

    return Q, R


def test_random_matrices():
    """Test QR factorization on random matrices of different sizes."""
    results = []

    for size in [5, 10, 15, 20, 25]:
        A = np.random.randn(size, size)

        Q_custom, R_custom = gram_schmidt_qr(A)
        Q_numpy, R_numpy = qr(A)

        error_Q = norm(Q_custom @ Q_custom.T - np.eye(size))
        error_Q_np = norm(Q_numpy @ Q_numpy.T - np.eye(size))
        error_reconstruction = norm(A - Q_custom @ R_custom)
        error_reconstruction_np = norm(A - Q_numpy @ R_numpy)
        error_vs_numpy_Q = norm(abs(Q_custom) - abs(Q_numpy))
        error_vs_numpy_R = norm(abs(R_custom) - abs(R_numpy))

        results.append(
            {
                "size": size,
                "error_orthogonality": error_Q,
                "err_ort_np": error_Q_np,
                "error_reconstruction": error_reconstruction,
                "err_rec_np": error_reconstruction_np,
                "error_vs_numpy_Q": error_vs_numpy_Q,
                "error_vs_numpy_R": error_vs_numpy_R,
                "cond_A": cond(A),
            }
        )

    return results


def generate_condition_matrices():
    """Generate matrices with varying condition numbers for stability analysis."""
    matrices = []

    for i in range(30):
        U, S, V = np.linalg.svd(np.random.randint(-10, 10, size = (8, 8)))
        S = np.linspace(1, 1+i*10, 8)
        A = U @ np.diag(S) @ V
        matrices.append(A)

    return matrices


def analyze_qr_stability(matrices):
    """Analyze the stability of QR factorization on matrices with different condition numbers."""
    results = []

    for A in matrices:
        Q_custom, _ = gram_schmidt_qr(A)

        error_orthogonality = norm(Q_custom @ Q_custom.T - np.eye(A.shape[0]))
        condition_A = cond(A)

        results.append(
            {
                "condition_number": condition_A,
                "orthogonality_error": error_orthogonality,
            }
        )

    return results


def visualize_results(stability_results):
    """Visualize results of stability analysis."""
    plt.figure(figsize=(8, 8))

    cond_nums = [r["condition_number"] for r in stability_results]
    orth_errors = [r["orthogonality_error"] for r in stability_results]
    plt.plot(cond_nums, orth_errors)
    plt.xlabel("Condition number")
    plt.ylabel("Orthogonality error ||I - Q^T Q||")
    plt.title("Stability vs conditioning")
    plt.grid(True)

    plt.show()


def main():
    """Main function to run the QR factorization tests and analysis."""
    print("Test on random matrices:")
    random_results = test_random_matrices()

    for result in random_results:
        print(f"{result['size']}x{result['size']} matrix:")
        print(f"  Q orthogonality error: {result['error_orthogonality']:.2e} (Numpy: {result['err_ort_np']:.2e})")
        print(f"  A reconstruction error: {result['error_reconstruction']:.2e} (Numpy: {result['err_rec_np']:.2e})")
        print(f"  Difference vs NumPy Q: {result['error_vs_numpy_Q']:.2e}")
        print(f"  Difference vs NumPy R: {result['error_vs_numpy_R']:.2e}")
        print(f"  Condition number of A: {result['cond_A']:.2f}")
        print()

    matrices = generate_condition_matrices()
    stability_results = analyze_qr_stability(matrices)

    visualize_results(stability_results)


if __name__ == "__main__":
    main()
