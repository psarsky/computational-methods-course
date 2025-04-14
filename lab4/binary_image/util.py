"""Utility and visualization functions for binary image processing."""
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.colors import ListedColormap


def generate_binary_image(n, density):
    """
    Generates a random binary image of size n x n with a given density of black pixels.
    """
    random_values = np.random.random((n, n))
    binary_image = np.zeros((n, n), dtype=int)
    binary_image[random_values < density] = 1
    return binary_image


def get_neighbors(i, j, n, neighborhood_type):
    """
    Returns the indices of neighbors of pixel (i, j) according to the selected neighborhood type.
    """
    neighbors = []

    # Up, down, left, right
    four_neighbors = [
        (i-1, j), (i+1, j), (i, j-1), (i, j+1)
    ]

    # Above + corners
    eight_neighbors = four_neighbors + [
        (i-1, j-1), (i-1, j+1), (i+1, j-1), (i+1, j+1)
    ]

    # Above + further pixels
    sixteen_neighbors = eight_neighbors + [
        (i-2, j), (i+2, j), (i, j-2), (i, j+2),
        (i-2, j-1), (i-2, j+1), (i+2, j-1), (i+2, j+1),
        (i-1, j-2), (i+1, j-2), (i-1, j+2), (i+1, j+2)
    ]

    if neighborhood_type == '4':
        potential_neighbors = four_neighbors
    elif neighborhood_type == '8':
        potential_neighbors = eight_neighbors
    elif neighborhood_type == '8-16':
        potential_neighbors = sixteen_neighbors
    else:
        raise ValueError("Unknown neighborhood type")

    # Neighbors within bounds
    for ni, nj in potential_neighbors:
        if 0 <= ni < n and 0 <= nj < n:
            neighbors.append((ni, nj))

    return neighbors


def visualize_results(original_image, optimized_image, energy_history, title):
    """
    Visualizes the results of simulated annealing.
    """
    _, axs = plt.subplots(1, 3, figsize=(18, 6))

    cmap = ListedColormap(['white', 'black'])

    axs[0].imshow(original_image, cmap=cmap)
    axs[0].set_title('Original image')
    axs[0].axis('off')

    axs[1].imshow(optimized_image, cmap=cmap)
    axs[1].set_title('Optimized image')
    axs[1].axis('off')

    axs[2].plot(energy_history)
    axs[2].set_title('Energy through iterations')
    axs[2].set_xlabel('Iteration')
    axs[2].set_ylabel('Energy')

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def animate(filename, images, energy_function, neighborhood_type, framerate=30):
    """Create an animation of image processing."""
    print("Creating animation...")

    best_energy = float('inf')
    cmap = ListedColormap(['white', 'black'])

    def update(frame):
        nonlocal best_energy
        current_energy = energy_function(frame, neighborhood_type)
        best_energy = min(best_energy, current_energy)

        ax.clear()
        ax.set_title(f"Current energy: {current_energy:.2f}, Best energy: {best_energy:.2f}")
        ax.set_xticks([])
        ax.set_yticks([])

        ax.imshow(frame, cmap=cmap)
        ax.legend()

    fig, ax = plt.subplots(figsize=(8, 8))
    step = len(images) // 200
    fr = images[::step]

    anim = animation.FuncAnimation(
        fig, update, frames=fr, interval=1000//framerate, repeat=False
    )

    directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "animations")
    os.makedirs(directory, exist_ok=True)
    file_path = os.path.join(directory, filename)

    with open(file_path, "w", encoding="utf-8") as file:
        file.write(anim.to_jshtml(fps=framerate))

    print(f"Saved animation to {filename}\n")

    plt.close(fig)
