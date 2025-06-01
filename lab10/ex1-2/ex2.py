"""Signal generation and FFT analysis."""

import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft, fftfreq


def generate_signal_a(fs=1000, duration=1, frequencies=None):
    """Generates a signal composed of a sum of sine waves with given frequencies.

    Args:
        fs (int): Sampling frequency in Hz.
        duration (float): Duration of the signal in seconds.
        frequencies (list): List of frequencies to include in the signal.

    Returns:
        tuple: Time array and generated signal array (t, signal).
    """
    if frequencies is None:
        frequencies = [50, 120, 200]
    t = np.linspace(0, duration, fs * duration, endpoint=False)
    signal = np.zeros_like(t)

    for freq in frequencies:
        amplitude = 1.0 / len(frequencies)
        signal += amplitude * np.sin(2 * np.pi * freq * t)

    return t, signal


def generate_signal_b(fs=1000, duration=1, base_frequencies=None, intervals=5):
    """Generates a signal with changing frequencies over time intervals.

    Args:
        fs (int): Sampling frequency in Hz.
        duration (float): Duration of the signal in seconds.
        base_frequencies (list): List of base frequencies to cycle through.
        intervals (int): Number of time intervals to divide the signal into.

    Returns:
        tuple: Time array and generated signal array (t, signal).
    """
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


def analyze_fft_results(X, fs):
    """Analyzes FFT results and extracts frequency domain information.

    Args:
        X (np.ndarray): Complex FFT results.
        fs (int): Sampling frequency in Hz.

    Returns:
        tuple: Frequencies, magnitude, phase, real part, and imaginary part arrays.
    """
    n = len(X)
    frequencies = fftfreq(n, 1 / fs)

    magnitude = np.abs(X)
    phase = np.angle(X)

    real_part = X.real
    imaginary_part = X.imag

    return frequencies, magnitude, phase, real_part, imaginary_part


def plot_signal_analysis(t, signal, X, fs, title):
    """Plots signal analysis results.

    Args:
        t (np.ndarray): Time array.
        signal (np.ndarray): Time-domain signal.
        X (np.ndarray): Complex FFT results.
        fs (int): Sampling frequency in Hz.
        title (str): Title for the plot.
    """
    frequencies, magnitude, phase, real_part, imaginary_part = analyze_fft_results(
        X, fs
    )

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(title, fontsize=16)

    axes[0, 0].plot(t, signal)
    axes[0, 0].set_title("Analyzed signal")
    axes[0, 0].set_xlabel("Time [s]")
    axes[0, 0].set_ylabel("Amplitude")
    axes[0, 0].grid(True)

    axes[0, 1].plot(
        frequencies[: len(frequencies) // 2], magnitude[: len(magnitude) // 2]
    )
    axes[0, 1].set_title("Amplitude spectrum")
    axes[0, 1].set_xlabel("Frequency [Hz]")
    axes[0, 1].set_ylabel("|X(f)|")
    axes[0, 1].grid(True)

    axes[0, 2].plot(frequencies[: len(frequencies) // 2], phase[: len(phase) // 2])
    axes[0, 2].set_title("Phase spectrum")
    axes[0, 2].set_xlabel("Frequency [Hz]")
    axes[0, 2].set_ylabel("Phase [rad]")
    axes[0, 2].grid(True)

    axes[1, 0].plot(
        frequencies[: len(frequencies) // 2], real_part[: len(real_part) // 2]
    )
    axes[1, 0].set_title("Real part")
    axes[1, 0].set_xlabel("Frequency [Hz]")
    axes[1, 0].set_ylabel("Re{X(f)}")
    axes[1, 0].grid(True)

    axes[1, 1].plot(
        frequencies[: len(frequencies) // 2], imaginary_part[: len(imaginary_part) // 2]
    )
    axes[1, 1].set_title("Imaginary part")
    axes[1, 1].set_xlabel("Frequency [Hz]")
    axes[1, 1].set_ylabel("Im{X(f)}")
    axes[1, 1].grid(True)

    axes[1, 2].scatter(real_part, imaginary_part, alpha=0.6, s=10)
    axes[1, 2].set_title("Complex plane")
    axes[1, 2].set_xlabel("Re{X(f)}")
    axes[1, 2].set_ylabel("Im{X(f)}")
    axes[1, 2].grid(True)
    axes[1, 2].axis("equal")

    plt.tight_layout()
    plt.show()


def main():
    """Main function that demonstrates signal generation and FFT analysis."""
    fs = 512
    duration = 1

    t_a, signal_a = generate_signal_a(fs, duration, [10, 20, 50, 120, 200])
    X_a = fft(signal_a)
    plot_signal_analysis(t_a, signal_a, X_a, fs, "Signal A: Sine sum")

    t_b, signal_b = generate_signal_b(fs, duration, [10, 20, 50, 120, 200], 5)
    X_b = fft(signal_b)
    plot_signal_analysis(t_b, signal_b, X_b, fs, "Signal B: Changing frequencies")


if __name__ == "__main__":
    main()
