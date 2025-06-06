"""Excercise 2: Simulated Annealing for Binary Image Optimization"""
import time

import numpy as np
from matplotlib import rcParams
from energy_functions import energy_clusters, energy_distance, energy_vertical_neighbors
from util import animate, generate_binary_image, visualize_results

rcParams['animation.embed_limit'] = 200


def simulated_annealing(image, neighborhood_type, energy_function=energy_clusters,
                        temperature=100.0, cooling_rate=0.999, threshold=1e-8):
    """
    Performs simulated annealing process for a binary image.
    """
    n = image.shape[0]
    current_energy = energy_function(image, neighborhood_type)

    best_image = image.copy()
    best_energy = current_energy

    energy_history = [current_energy]

    while temperature > threshold:
        x = np.random.randint(0, n), np.random.randint(0, n)
        y = np.random.randint(0, n), np.random.randint(0, n)
        while image[x] == image[y]:
            y = np.random.randint(0, n), np.random.randint(0, n)

        new_image = image.copy()
        new_image[x], new_image[y] = new_image[y], new_image[x]

        new_energy = energy_function(new_image, neighborhood_type)
        delta_energy = new_energy - current_energy

        if delta_energy < 0 or np.random.random() < np.exp(-delta_energy / temperature):
            image = new_image.copy()
            current_energy = new_energy

            if current_energy < best_energy:
                best_image = image.copy()
                best_energy = current_energy

        temperature *= cooling_rate
        energy_history.append(current_energy)

    return best_image, energy_history


def simulated_annealing_vis(image, neighborhood_type, energy_function=energy_clusters,
                        temperature=100.0, cooling_rate=0.999, threshold=1e-8):
    """
    Performs simulated annealing process and collects data for animation.
    """
    n = image.shape[0]
    current_energy = energy_function(image, neighborhood_type)
    best_energy = current_energy

    images = [image.copy()]

    i = 0
    while temperature > threshold:
        x = np.random.randint(0, n), np.random.randint(0, n)
        y = np.random.randint(0, n), np.random.randint(0, n)
        while image[x] == image[y]:
            y = np.random.randint(0, n), np.random.randint(0, n)

        new_image = image.copy()
        new_image[x], new_image[y] = new_image[y], new_image[x]

        new_energy = energy_function(new_image, neighborhood_type)
        delta_energy = new_energy - current_energy

        if delta_energy < 0 or np.random.random() < np.exp(-delta_energy / temperature):
            image = new_image.copy()
            current_energy = new_energy

            best_energy = min(best_energy, current_energy)

        images.append(image.copy())
        temperature *= cooling_rate
        i += 1

    return images


def main():
    """
    Main function to run the experiments and visualize results.
    """
    n = 50
    density = 0.4
    neighborhood_types = ['4', '8', '8-16']
    cooling_rates = [0.99, 0.995, 0.999]
    energy_functions = {
        'clusters': energy_clusters,
        'distance': energy_distance,
        'vertical_neighbors': energy_vertical_neighbors
    }

    original_image = generate_binary_image(n, density)

    for neighborhood_type in neighborhood_types:
        for rate in cooling_rates:
            for energy_fn_name, energy_fn in energy_functions.items():
                start_time = time.time()
                optimized_image, energy_history = simulated_annealing(
                    original_image,
                    neighborhood_type,
                    energy_function=energy_fn,
                    cooling_rate=rate
                )
                elapsed_time = time.time() - start_time

                print(f"Neighborhood: {neighborhood_type}, Cooling rate: {rate}, Energy function: {energy_fn_name}")
                print(f"Initial energy: {energy_fn(original_image, neighborhood_type)}")
                print(f"Final energy: {energy_fn(optimized_image, neighborhood_type)}")
                print(f"Total elapsed time: {elapsed_time:.2f} s\n")

                visualize_results(original_image, optimized_image, energy_history,
                        f"Neighborhood: {neighborhood_type}, Cooling rate: {rate}, Energy function: {energy_fn_name}")

    original_image = generate_binary_image(100, 0.4)
    for energy_fn_name, energy_fn in energy_functions.items():
        images = simulated_annealing_vis(
            original_image,
            '8',
            energy_function=energy_fn,
            cooling_rate=0.999,
            threshold=1e-5
        )

        animate(f"{energy_fn_name}.html", images, energy_fn, '8', framerate=30)


if __name__ == "__main__":
    main()
