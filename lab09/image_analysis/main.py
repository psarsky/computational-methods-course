"""Fourier transform and pattern detection in images"""

import os

import numpy as np
from skimage.feature import peak_local_max
from util import (load_and_preprocess_image, visualize_fourier_transformation,
                  visualize_results)


def analyze_fourier_transform(image):
    """Perform Fourier transform analysis on the image.

    Args:
        image (np.ndarray): Input image.

    Returns:
        tuple: Magnitude and phase values of the Fourier transform.
    """
    fft_image = np.fft.fft2(image)
    fft_shifted = np.fft.fftshift(fft_image)

    magnitude = np.abs(fft_shifted)
    phase = np.angle(fft_shifted)

    magnitude_log = np.log(magnitude + 1)

    return magnitude_log, phase


def detect_pattern_correlation(image, pattern):
    """Detects the correlation between the image and the pattern using Fourier transform.

    Args:
        image (np.ndarray): Input image.
        pattern (np.ndarray): Pattern to detect.

    Returns:
        np.ndarray: Correlation map.
    """
    ph, pw = pattern.shape

    pattern_rotated = np.rot90(pattern, 2)

    pattern_padded = np.zeros_like(image)
    pattern_padded[:ph, :pw] = pattern_rotated

    fft_image = np.fft.fft2(image)
    fft_pattern = np.fft.fft2(pattern_padded)

    correlation_fft = fft_image * fft_pattern

    correlation = np.real(np.fft.ifft2(correlation_fft))

    return correlation


def find_pattern_locations(correlation, threshold_percentile=95, min_distance=10):
    """Finds the locations of the detected patterns in the correlation map.

    Args:
        correlation (np.ndarray): Correlation map.
        threshold_percentile (int, optional): Percentile threshold for peak detection. Defaults to 95.
        min_distance (int, optional): Minimum distance between detected peaks. Defaults to 10.

    Returns:
        tuple: List of detected pattern locations and the threshold value.
    """
    threshold = np.percentile(correlation, threshold_percentile)

    peaks = peak_local_max(
        correlation, min_distance=min_distance, threshold_abs=threshold
    )

    return list(peaks), threshold


def main():
    """Main function to perform image analysis."""
    IMG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")
    os.makedirs(IMG_PATH, exist_ok=True)

    datasets = [
        {
            "name": "Text (galia)",
            "image_path": os.path.join(IMG_PATH, "galia.png"),
            "pattern_path": os.path.join(IMG_PATH, "galia_e.png"),
            "invert_colors": True,
            "threshold_percentile": 99.9902,  # best threshold value (found by trial and error)
        },  # which detects all 'e' characters without false positives
        {
            "name": "Fish school",
            "image_path": os.path.join(IMG_PATH, "school.jpg"),
            "pattern_path": os.path.join(IMG_PATH, "fish1.png"),
            "invert_colors": False,
            "threshold_percentile": 90,
        },
    ]

    for dataset in datasets:
        print(f"Image analysis: {dataset['name']}\n")

        main_image = load_and_preprocess_image(
            dataset["image_path"], dataset["invert_colors"]
        )
        pattern_image = load_and_preprocess_image(
            dataset["pattern_path"], dataset["invert_colors"]
        )

        if main_image is None or pattern_image is None:
            print(f"Error reading images for dataset {dataset['name']}")
            continue

        print("\n1. Fourier transformation analysis")
        magnitude_log, phase = analyze_fourier_transform(main_image)
        visualize_fourier_transformation(
            magnitude_log, phase, main_image, dataset["name"]
        )

        print("\n2. Pattern detection")
        correlation = np.abs(detect_pattern_correlation(main_image, pattern_image))

        locations, threshold = find_pattern_locations(
            correlation, threshold_percentile=dataset["threshold_percentile"]
        )

        print(f"\nDetection threshold: {threshold:.4f}")
        print(f"Pattern occurrences: {len(locations)}")

        visualize_results(
            main_image,
            pattern_image,
            correlation,
            locations,
            dataset["name"],
            pattern_shape=pattern_image.shape,
        )

        print("\nCorrelation statistics:")
        print(f"  Maximum value: {np.max(correlation):.4f}")
        print(f"  Minimum value: {np.min(correlation):.4f}")
        print(f"  Average value: {np.mean(correlation):.4f}")
        print(f"  Standard deviation: {np.std(correlation):.4f}")

        print(f"\n{'='*40}\n")


if __name__ == "__main__":
    main()
