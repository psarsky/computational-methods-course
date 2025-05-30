import math
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.fft import fft, ifft


def dft(x):
    n = len(x)
    X = np.zeros(n, dtype=complex)

    for k in range(n):
        for j in range(n):
            angle = -2 * math.pi * k * j / n
            X[k] += x[j] * (math.cos(angle) + 1j * math.sin(angle))

    return X


def idft(X):
    n = len(X)
    x = np.zeros(n, dtype=complex)

    for j in range(n):
        for k in range(n):
            angle = 2 * math.pi * k * j / n
            x[j] += X[k] * (math.cos(angle) + 1j * math.sin(angle))
        x[j] /= n

    return x


def fft_cooley_tukey(x):
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
    x = [1, 2, 3, 4]
    np.set_printoptions(precision=2, suppress=False)

    print("Wektor wejściowy:", x)

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

    print("Różnica DFT vs NumPy:", np.max(np.abs(np.array(X_dft) - X_numpy)))
    print("Różnica IDFT vs NumPy:", np.max(np.abs(np.array(x_recovered) - x_rec_np)))
    print("Różnica FFT vs NumPy:", np.max(np.abs(np.array(X_fft) - X_numpy)))
    print(f"Czas DFT: {time_dft:.6f}s")
    print(f"Czas FFT: {time_fft:.6f}s")
    print(f"Czas NumPy FFT: {time_np:.6f}s")


def benchmark_implementations():
    sizes = [4, 8, 16, 32, 64, 128, 256, 512]
    times_dft = []
    times_fft = []
    times_numpy = []
    errors_dft = []
    errors_fft = []

    for size in sizes:
        print(f"Testowanie rozmiaru: {size}")

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
                "Rozmiar": size,
                "DFT [s]": f"{times_dft[i]:.6f}",
                "FFT [s]": f"{times_fft[i]:.6f}",
                "NumPy FFT [s]": f"{times_numpy[i]:.6f}",
                "Błąd DFT": f"{errors_dft[i]:.2e}",
                "Błąd FFT": f"{errors_fft[i]:.2e}",
                "Przyspieszenie FFT": f"{speedup:.1f}x",
            }
        )

    df = pd.DataFrame(results)
    print("\nWyniki benchmarku:")
    print(df.to_string(index=False))

    return df


def plot_benchmark_results(
    sizes, times_dft, times_fft, times_numpy, errors_dft, errors_fft
):
    """Wizualizacja wyników benchmarku."""

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("Porównanie implementacji FFT", fontsize=16, fontweight="bold")

    axes[0, 0].plot(sizes, times_dft, "o-", label="DFT", linewidth=2, markersize=6)
    axes[0, 0].plot(
        sizes, times_fft, "s-", label="FFT Cooley-Tukey", linewidth=2, markersize=6
    )
    axes[0, 0].plot(
        sizes, times_numpy, "^-", label="NumPy FFT", linewidth=2, markersize=6
    )
    axes[0, 0].set_xlabel("Rozmiar danych")
    axes[0, 0].set_ylabel("Czas wykonania [s]")
    axes[0, 0].set_title("Czasy wykonania (skala liniowa)")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].loglog(sizes, times_dft, "o-", label="DFT", linewidth=2, markersize=6)
    axes[0, 1].loglog(
        sizes, times_fft, "s-", label="FFT Cooley-Tukey", linewidth=2, markersize=6
    )
    axes[0, 1].loglog(
        sizes, times_numpy, "^-", label="NumPy FFT", linewidth=2, markersize=6
    )
    axes[0, 1].set_xlabel("Rozmiar danych")
    axes[0, 1].set_ylabel("Czas wykonania [s]")
    axes[0, 1].set_title("Czasy wykonania (skala logarytmiczna)")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    speedup = [times_dft[i] / times_fft[i] for i in range(len(sizes))]
    axes[1, 0].plot(sizes, speedup, "ro-", linewidth=2, markersize=6)
    axes[1, 0].set_xlabel("Rozmiar danych")
    axes[1, 0].set_ylabel("Przyspieszenie [razy]")
    axes[1, 0].set_title("Przyspieszenie FFT względem DFT")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_yscale("log")

    axes[1, 1].semilogy(
        sizes, errors_dft, "o-", label="Błąd DFT", linewidth=2, markersize=6
    )
    axes[1, 1].semilogy(
        sizes, errors_fft, "s-", label="Błąd FFT", linewidth=2, markersize=6
    )
    axes[1, 1].set_xlabel("Rozmiar danych")
    axes[1, 1].set_ylabel("Maksymalny błąd")
    axes[1, 1].set_title("Błędy numeryczne względem NumPy")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout(pad=3.0)
    plt.show()


if __name__ == "__main__":
    test_implementations()
    print("\n" + "="*60)
    benchmark_implementations()
