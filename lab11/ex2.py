"""Polynomial approximation using QR factorization"""

import matplotlib.pyplot as plt
import numpy as np
from ex1 import gram_schmidt_qr
from numpy.linalg import lstsq


def solve(A, b):
    """Solve the overdetermined system using QR factorization."""
    Q, R = gram_schmidt_qr(A)
    return np.linalg.inv(R) @ Q.T @ b


def polynomial_approximation():
    """Polynomial approximation using QR method and comparison with NumPy's lstsq."""
    x_data = np.array([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
    y_data = np.array([2, 7, 9, 12, 13, 14, 14, 13, 10, 8, 4])

    A = np.column_stack([np.ones(len(x_data)), x_data, x_data**2])

    coeffs_qr = solve(A, y_data)

    coeffs_numpy = lstsq(A, y_data, rcond=None)[0]

    print("Polynomial approximation using QR method\n")
    print("Model: f(x) = α₀ + α₁x + α₂x²")
    print()

    print("QR method coefficients:")
    print(f"α₀ = {coeffs_qr[0]:.6f}")
    print(f"α₁ = {coeffs_qr[1]:.6f}")
    print(f"α₂ = {coeffs_qr[2]:.6f}")
    print()

    print("NumPy lstsq coefficients:")
    print(f"α₀ = {coeffs_numpy[0]:.6f}")
    print(f"α₁ = {coeffs_numpy[1]:.6f}")
    print(f"α₂ = {coeffs_numpy[2]:.6f}")
    print()

    diff = np.abs(coeffs_qr - coeffs_numpy)
    print("Difference between methods:")
    print(f"Δα₀ = {diff[0]:.2e}")
    print(f"Δα₁ = {diff[1]:.2e}")
    print(f"Δα₂ = {diff[2]:.2e}")
    print()

    residuals_qr = A @ coeffs_qr - y_data
    residuals_numpy = A @ coeffs_numpy - y_data

    rms_qr = np.sqrt(np.mean(residuals_qr**2))
    rms_numpy = np.sqrt(np.mean(residuals_numpy**2))

    print(f"RMS error (QR method): {rms_qr:.6f}")
    print(f"RMS error (NumPy lstsq): {rms_numpy:.6f}")

    x_plot = np.linspace(-6, 6, 200)
    y_qr = coeffs_qr[0] + coeffs_qr[1] * x_plot + coeffs_qr[2] * x_plot**2
    y_numpy = coeffs_numpy[0] + coeffs_numpy[1] * x_plot + coeffs_numpy[2] * x_plot**2

    visualize_polynomial_fit(
        x_data,
        y_data,
        x_plot,
        y_qr,
        y_numpy
    )


def visualize_polynomial_fit(
    x_data,
    y_data,
    x_plot,
    y_qr,
    y_numpy
):
    """Visualize polynomial fit and residuals."""
    plt.figure(figsize=(8, 6))

    plt.plot(x_data, y_data, "ro", markersize=8, label="Data points")
    plt.plot(x_plot, y_qr, "b-", linewidth=2, label="QR method")
    plt.plot(x_plot, y_numpy, "g--", linewidth=2, label="NumPy lstsq")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Polynomial Approximation")
    plt.legend()
    plt.grid(True)

    plt.show()


if __name__ == "__main__":
    polynomial_approximation()
