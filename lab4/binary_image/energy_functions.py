"""Energy functions for binary image processing."""
import numpy as np
from util import get_neighbors


def energy_clusters(image, neighborhood_type):
    """
    Calculates the energy of the image (neighboring pixels of the same value decrease energy).
    """
    n = image.shape[0]
    energy = 0

    for i in range(n):
        for j in range(n):
            neighbors = get_neighbors(i, j, n, neighborhood_type)
            for ni, nj in neighbors:
                if image[i, j] != image[ni, nj]:
                    energy += 1.0

    return energy


def energy_vertical_neighbors(image, neighborhood_type):
    """
    Calculates the energy of the image (vertical neighboring pixels change energy).
    """
    n = image.shape[0]
    energy = 0

    for i in range(n):
        for j in range(n):
            neighbors = get_neighbors(i, j, n, neighborhood_type)
            for ni, nj in neighbors:
                if (j != nj) and image[i, j] == image[ni, nj]:
                    energy += 1.0

    return energy


def energy_ising(image, neighborhood_type):
    """
    Calculates the energy of the image according to an adaptation of the Ising model.
    """
    n = image.shape[0]
    energy = 0

    for i in range(n):
        for j in range(n):
            pixel_value = image[i, j]
            neighbors = get_neighbors(i, j, n, neighborhood_type)

            for ni, nj in neighbors:
                neighbor_value = image[ni, nj]

                distance = np.sqrt((i - ni)**2 + (j - nj)**2)

                if pixel_value != neighbor_value:
                    energy += 1.0 / distance

    return energy
