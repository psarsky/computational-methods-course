"""DFT, IDFT, and FFT implementations and tests."""

import math
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.fft import fft, ifft


def dft(x):
    """Computes the Discrete Fourier Transform (DFT) of a 1D array.

    Args:
        x (list): Input array of numbers.

    Returns:
        np.ndarray: DFT of the input array.
    """
    n = len(x)
    X = np.zeros(n, dtype=complex)

    for k in range(n):
        for j in range(n):
            angle = -2 * math.pi * k * j / n
            X[k] += x[j] * (math.cos(angle) + 1j * math.sin(angle))

    return X


def idft(X):
    """Computes the Inverse Discrete Fourier Transform (IDFT) of a 1D array.

    Args:
        X (np.ndarray): Input array of complex numbers representing the DFT.

    Returns:
        np.ndarray: IDFT of the input array.
    """
    n = len(X)
    x = np.zeros(n, dtype=complex)

    for j in range(n):
        for k in range(n):
            angle = 2 * math.pi * k * j / n
            x[j] += X[k] * (math.cos(angle) + 1j * math.sin(angle))
        x[j] /= n

    return x


def fft_cooley_tukey(x):
    """Computes the Fast Fourier Transform (FFT) using the Cooley-Tukey algorithm.

    Args:
        x (list): Input array of numbers.

    Returns:
        list: FFT of the input list.
    """
    n = len(x)

    if n <= 1:
        return x

    even = fft_cooley_tukey(x[::2])
    odd = fft_cooley_tukey(x[1::2])

    T = [
        math.cos(-2 * math.pi * k / n) + 1j * math.sin(-2 * math.pi * k / n)
        for k in range(n // 2)
    ]

    return [even[k] + T[k] * odd[k] for k in range(n // 2)] + [
        even[k] - T[k] * odd[k] for k in range(n // 2)
    ]


def test_implementations():
    """Tests the DFT, IDFT, and FFT implementations against NumPy's FFT."""
    x = [1, 2, 3, 4]
    np.set_printoptions(precision=2, suppress=False)

    print("Input vector:", x)
    print()

    time_dft = time.time()
    X_dft = dft(x)
    time_dft = time.time() - time_dft
    print("DFT:", X_dft)

    x_recovered = idft(X_dft)
    print("IDFT:", x_recovered)

    x_rec_np = ifft(X_dft)

    x_padded = x + [0] * (4 - len(x) % 4) if len(x) % 4 != 0 else x
    time_fft = time.time()
    X_fft = fft_cooley_tukey(x_padded)
    time_fft = time.time() - time_fft
    print("FFT:", X_fft)

    time_np = time.time()
    X_numpy = fft(x)
    time_np = time.time() - time_np
    print("NumPy FFT:", X_numpy)
    print()

    print(f"DFT vs NumPy difference:  {np.max(np.abs(np.array(X_dft) - X_numpy)):.5e}")
    print(
        f"IDFT vs NumPy difference: {np.max(np.abs(np.array(x_recovered) - x_rec_np)):.5e}"
    )
    print(f"FFT vs NumPy difference:  {np.max(np.abs(np.array(X_fft) - X_numpy)):.5e}")
    print()
    print(f"DFT time:       {time_dft:.6f}s")
    print(f"FFT time:       {time_fft:.6f}s")
    print(f"NumPy FFT time: {time_np:.6f}s")


def benchmark_implementations():
    """Benchmarks the DFT, FFT, and NumPy FFT implementations for various input sizes."""
    sizes = [4, 8, 16, 32, 64, 128, 256, 512]
    times_dft = []
    times_fft = []
    times_numpy = []
    errors_dft = []
    errors_fft = []

    for size in sizes:
        x = np.random.random(size).tolist()

        start_time = time.time()
        X_dft = dft(x)
        time_dft = time.time() - start_time
        times_dft.append(time_dft)

        start_time = time.time()
        X_fft = fft_cooley_tukey(x)
        time_fft = time.time() - start_time
        times_fft.append(time_fft)

        start_time = time.time()
        X_numpy = fft(np.array(x))
        time_numpy = time.time() - start_time
        times_numpy.append(time_numpy)

        error_dft = np.max(np.abs(np.array(X_dft) - X_numpy))
        error_fft = np.max(np.abs(np.array(X_fft) - X_numpy))
        errors_dft.append(error_dft)
        errors_fft.append(error_fft)

    plot_benchmark_results(
        sizes, times_dft, times_fft, times_numpy, errors_dft, errors_fft
    )

    results = []
    for i, size in enumerate(sizes):
        speedup = times_dft[i] / times_fft[i] if times_fft[i] > 0 else 0
        results.append(
            {
                "Size": size,
                "DFT [s]": f"{times_dft[i]:.6f}",
                "FFT [s]": f"{times_fft[i]:.6f}",
                "NumPy FFT [s]": f"{times_numpy[i]:.6f}",
                "DFT error": f"{errors_dft[i]:.2e}",
                "FFT error": f"{errors_fft[i]:.2e}",
                "FFT speedup": f"{speedup:.1f}x",
            }
        )

    df = pd.DataFrame(results)
    print("\nBenchmark results:")
    print(df.to_string(index=False))


def plot_benchmark_results(
    sizes, times_dft, times_fft, times_numpy, errors_dft, errors_fft
):
    """Plots the benchmark results of DFT, FFT, and NumPy FFT implementations.

    Args:
        sizes (list): Sizes of the input vectors.
        times_dft (list): Execution times for the DFT implementation.
        times_fft (list): Execution times for the FFT implementation.
        times_numpy (list): Execution times for NumPy's FFT implementation.
        errors_dft (list): Maximum errors for the DFT implementation.
        errors_fft (list): Maximum errors for the FFT implementation.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("FFT implementation comparison", fontsize=16, fontweight="bold")

    axes[0, 0].plot(sizes, times_dft, "o-", label="DFT", linewidth=2, markersize=6)
    axes[0, 0].plot(
        sizes, times_fft, "s-", label="FFT Cooley-Tukey", linewidth=2, markersize=6
    )
    axes[0, 0].plot(
        sizes, times_numpy, "^-", label="NumPy FFT", linewidth=2, markersize=6
    )
    axes[0, 0].set_xlabel("Vector size")
    axes[0, 0].set_ylabel("Execution time [s]")
    axes[0, 0].set_title("Execution times (linear scale)")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].loglog(sizes, times_dft, "o-", label="DFT", linewidth=2, markersize=6)
    axes[0, 1].loglog(
        sizes, times_fft, "s-", label="FFT Cooley-Tukey", linewidth=2, markersize=6
    )
    axes[0, 1].loglog(
        sizes, times_numpy, "^-", label="NumPy FFT", linewidth=2, markersize=6
    )
    axes[0, 1].set_xlabel("Vector size")
    axes[0, 1].set_ylabel("Execution time [s]")
    axes[0, 1].set_title("Execution times (log scale)")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    speedup = [times_dft[i] / times_fft[i] for i in range(len(sizes))]
    axes[1, 0].plot(sizes, speedup, "ro-", linewidth=2, markersize=6)
    axes[1, 0].set_xlabel("Vector size")
    axes[1, 0].set_ylabel("Factor")
    axes[1, 0].set_title("Speedup of FFT over DFT")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].semilogy(
        sizes, errors_dft, "o-", label="DFT error", linewidth=2, markersize=6
    )
    axes[1, 1].semilogy(
        sizes, errors_fft, "s-", label="FFT error", linewidth=2, markersize=6
    )
    axes[1, 1].set_xlabel("Vector size")
    axes[1, 1].set_ylabel("Maximum error")
    axes[1, 1].set_title("Maximum error of DFT and FFT")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout(pad=3.0)
    plt.show()


if __name__ == "__main__":
    test_implementations()
    benchmark_implementations()
