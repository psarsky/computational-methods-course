"""Visualization and image loading utilities for image analysis."""

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def load_and_preprocess_image(image_path, invert_colors=False):
    """Load and preprocess an image.

    Args:
        image_path (str): Path to the image file.
        invert_colors (bool, optional): Whether to invert the colors of the image. Defaults to False.

    Returns:
        np.ndarray: Preprocessed image as a NumPy array.
    """
    try:
        img = Image.open(image_path)

        if img.mode != "L":
            img = img.convert("L")

        img_array = np.array(img, dtype=np.float64)

        if invert_colors:
            img_array = 255 - img_array

        img_array = img_array / 255.0

        return img_array
    except Exception as e:
        print(f"Error while processing {image_path}: {e}")
        return None


def visualize_fourier_transformation(magnitude_log, phase, image, title=""):
    """Visualize the values of magnitude and phase of the Fourier transformation of an image.

    Args:
        magnitude_log (np.ndarray): Array of magnitude values in log scale.
        phase (np.ndarray): Array of phase values.
        image (np.ndarray): Original image.
        title (str, optional): Title for the plot. Defaults to "".
    """
    fig = plt.figure(figsize=(10, 10))
    fig.suptitle(f"{title} - Fourier transformation analysis", fontsize=16, y=0.98)

    gs = fig.add_gridspec(
        2, 2, height_ratios=[1, 1], width_ratios=[1, 1], hspace=0.4, wspace=0.3
    )

    ax_top = fig.add_subplot(gs[0, :])
    ax_top.imshow(image, cmap="gray")
    ax_top.set_title("Original image", pad=10)
    ax_top.axis("off")

    ax_bottom_left = fig.add_subplot(gs[1, 0])
    ax_bottom_left.imshow(magnitude_log, cmap="gray")
    ax_bottom_left.set_title("Magnitude (log scale)", pad=10)
    ax_bottom_left.axis("off")

    ax_bottom_right = fig.add_subplot(gs[1, 1])
    ax_bottom_right.imshow(phase, cmap="hsv")
    ax_bottom_right.set_title("Phase", pad=10)
    ax_bottom_right.axis("off")

    plt.subplots_adjust(left=0.08, right=0.92, top=0.9, bottom=0.05)
    plt.show()


def visualize_results(
    original_image, pattern, correlation, locations, title_prefix="", pattern_shape=None
):
    """Visualize the results of pattern matching.

    Args:
        original_image (np.ndarray): Image to detect patterns in.
        pattern (np.ndarray): Pattern to detect.
        correlation (np.ndarray): Correlation map.
        locations (list): List of detected pattern locations (coordinates of lower right corners).
        title_prefix (str, optional): Prefix for the title of the plots. Defaults to "".
        pattern_shape (tuple, optional): Shape of the pattern (height, width). Defaults to None.
    """
    fig1, axes = plt.subplots(2, 2, figsize=(10, 10))
    fig1.suptitle(f"{title_prefix} - Pattern detection", fontsize=16)

    axes[0, 0].imshow(original_image, cmap="gray")
    if locations and pattern_shape is not None:
        ph, pw = pattern_shape
        for y, x in locations:
            rect = plt.Rectangle(
                (x - pw, y - ph), pw, ph, linewidth=1, edgecolor="r", facecolor="none"
            )
            axes[0, 0].add_patch(rect)
    axes[0, 0].set_title(f"Detected patterns ({len(locations)} found)", pad=10)
    axes[0, 0].axis("off")

    axes[0, 1].imshow(pattern, cmap="gray")
    axes[0, 1].set_title("Pattern")
    axes[0, 1].axis("off")

    axes[1, 0].imshow(correlation, cmap="hot")
    axes[1, 0].set_title("Correlation map", pad=10)
    axes[1, 0].axis("off")

    axes[1, 1].hist(correlation.flatten(), bins=100, alpha=0.7)
    axes[1, 1].set_title("Correlation histogram", pad=10)
    axes[1, 1].set_xlabel("Correlation value")
    axes[1, 1].set_ylabel("Frequency")

    plt.subplots_adjust(left=0.08, right=0.92, top=0.9, bottom=0.1)

    plt.show()

    fig2 = plt.figure(figsize=(8, 8))
    fig2.suptitle(f"{title_prefix} - 3D correlation surface", fontsize=16)

    ax_3d = fig2.add_subplot(111, projection="3d")
    y_3d, x_3d = np.mgrid[0 : correlation.shape[0] : 10, 0 : correlation.shape[1] : 10]
    ax_3d.plot_surface(x_3d, y_3d, correlation[::10, ::10], cmap="hot", alpha=0.7)

    plt.show()
