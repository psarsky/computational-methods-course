import math

import matplotlib.pyplot as plt
import numpy as np


def dft(x):
    n = len(x)
    X = np.zeros(n, dtype=complex)

    for k in range(n):
        for j in range(n):
            angle = -2 * math.pi * k * j / n
            X[k] += x[j] * (math.cos(angle) + 1j * math.sin(angle))

    return X


def generate_signal_a(fs=1000, duration=1, frequencies=None):
    if frequencies is None:
        frequencies = [50, 120, 200]
    t = np.linspace(0, duration, fs * duration, endpoint=False)
    signal = np.zeros_like(t)

    for freq in frequencies:
        amplitude = 1.0 / len(frequencies)
        signal += amplitude * np.sin(2 * np.pi * freq * t)

    return t, signal


def generate_signal_b(fs=1000, duration=1, base_frequencies=None, intervals=5):
    if base_frequencies is None:
        base_frequencies = [50, 120, 200]
    t = np.linspace(0, duration, fs * duration, endpoint=False)
    signal = np.zeros_like(t)

    samples_per_interval = len(t) // intervals

    for i in range(intervals):
        start_idx = i * samples_per_interval
        end_idx = (i + 1) * samples_per_interval if i < intervals - 1 else len(t)

        freq = base_frequencies[i % len(base_frequencies)]
        t_interval = t[start_idx:end_idx]

        signal[start_idx:end_idx] = np.sin(2 * np.pi * freq * t_interval)

    return t, signal


def analyze_dft_results(X, fs):
    n = len(X)
    frequencies = np.fft.fftfreq(n, 1 / fs)

    magnitude = np.abs(X)
    phase = np.angle(X)

    real_part = X.real
    imaginary_part = X.imag

    return frequencies, magnitude, phase, real_part, imaginary_part


def plot_signal_analysis(t, signal, X, fs, title):
    frequencies, magnitude, phase, real_part, imaginary_part = analyze_dft_results(
        X, fs
    )

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(title, fontsize=16)

    axes[0, 0].plot(t, signal)
    axes[0, 0].set_title("Sygnał czasowy")
    axes[0, 0].set_xlabel("Czas [s]")
    axes[0, 0].set_ylabel("Amplituda")
    axes[0, 0].grid(True)

    axes[0, 1].plot(
        frequencies[: len(frequencies) // 2], magnitude[: len(magnitude) // 2]
    )
    axes[0, 1].set_title("Widmo amplitudowe")
    axes[0, 1].set_xlabel("Częstotliwość [Hz]")
    axes[0, 1].set_ylabel("|X(f)|")
    axes[0, 1].grid(True)

    axes[0, 2].plot(frequencies[: len(frequencies) // 2], phase[: len(phase) // 2])
    axes[0, 2].set_title("Widmo fazowe")
    axes[0, 2].set_xlabel("Częstotliwość [Hz]")
    axes[0, 2].set_ylabel("Faza [rad]")
    axes[0, 2].grid(True)

    axes[1, 0].plot(
        frequencies[: len(frequencies) // 2], real_part[: len(real_part) // 2]
    )
    axes[1, 0].set_title("Część rzeczywista")
    axes[1, 0].set_xlabel("Częstotliwość [Hz]")
    axes[1, 0].set_ylabel("Re{X(f)}")
    axes[1, 0].grid(True)

    axes[1, 1].plot(
        frequencies[: len(frequencies) // 2], imaginary_part[: len(imaginary_part) // 2]
    )
    axes[1, 1].set_title("Część urojona")
    axes[1, 1].set_xlabel("Częstotliwość [Hz]")
    axes[1, 1].set_ylabel("Im{X(f)}")
    axes[1, 1].grid(True)

    axes[1, 2].scatter(real_part, imaginary_part, alpha=0.6, s=10)
    axes[1, 2].set_title("Płaszczyzna zespolona")
    axes[1, 2].set_xlabel("Re{X(f)}")
    axes[1, 2].set_ylabel("Im{X(f)}")
    axes[1, 2].grid(True)
    axes[1, 2].axis("equal")

    plt.tight_layout()


def main():
    fs = 512
    duration = 1

    print("=== ANALIZA SYGNAŁU A ===")
    t_a, signal_a = generate_signal_a(fs, duration, [50, 120, 200])

    X_a = dft(signal_a)

    frequencies, magnitude, _, _, _ = analyze_dft_results(X_a, fs)

    dominant_freqs = []
    threshold = np.max(magnitude) * 0.1
    for i, mag in enumerate(magnitude[: len(magnitude) // 2]):
        if mag > threshold and frequencies[i] > 0:
            dominant_freqs.append((frequencies[i], mag))

    print("Wykryte częstotliwości dominujące:")
    for freq, mag in sorted(dominant_freqs):
        print(f"  {freq:.1f} Hz: amplituda {mag:.3f}")

    print(f"Maksymalna amplituda: {np.max(magnitude):.3f}")

    print("\n=== ANALIZA SYGNAŁU B ===")
    t_b, signal_b = generate_signal_b(fs, duration, [50, 120, 200], 5)

    X_b = dft(signal_b)

    frequencies, magnitude, _, _, _ = analyze_dft_results(X_b, fs)

    dominant_freqs = []
    threshold = np.max(magnitude) * 0.05
    for i, mag in enumerate(magnitude[: len(magnitude) // 2]):
        if mag > threshold and frequencies[i] > 0:
            dominant_freqs.append((frequencies[i], mag))

    print("Wykryte częstotliwości dominujące:")
    for freq, mag in sorted(dominant_freqs)[:10]:
        print(f"  {freq:.1f} Hz: amplituda {mag:.3f}")

    print(f"Maksymalna amplituda: {np.max(magnitude):.3f}")

    plot_signal_analysis(t_a, signal_a, X_a, fs, "Sygnał A: Suma sinusoid")
    plot_signal_analysis(t_b, signal_b, X_b, fs, "Sygnał B: Sygnał złożony")

    plt.show()


if __name__ == "__main__":
    main()
