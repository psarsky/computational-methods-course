"""Excercise 1: Simulated annealing for TSP"""
import math
import random
import time

import matplotlib.pyplot as plt
import numpy as np


def simulated_annealing(points, temperature=100, cooling_rate=0.995, threshold=1e-2, swap_type="arbitrary"):
    """Solve the TSP using simulated annealing."""
    n = len(points)
    dist_matrix = distance_matrix(points)

    current_tour = list(range(n))
    random.shuffle(current_tour)
    current_length = tour_length(current_tour, dist_matrix)

    initial_tour = current_tour.copy()
    initial_length = current_length

    best_tour = current_tour.copy()
    best_length = current_length
    found_time = t_0 = time.perf_counter()

    lengths = [(0, initial_length)]
    temps = [(0, temperature)]

    while temperature > threshold:
        for _ in range(100):
            match swap_type:
                case "consecutive":
                    i = random.randint(0, n-2)
                    j = i + 1
                case "arbitrary":
                    i, j = random.sample(range(n), 2)

            new_tour = current_tour.copy()
            new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
            new_length = tour_length(new_tour, dist_matrix)

            if new_length < current_length or random.random() < math.exp((current_length - new_length) / temperature):
                current_tour = new_tour
                current_length = new_length

                if new_length < best_length:
                    best_tour = new_tour
                    best_length = new_length
                    found_time = time.perf_counter() - t_0

            lengths.append((time.perf_counter() - t_0, current_length))

        temperature *= cooling_rate
        temps.append((time.perf_counter() - t_0, temperature))

    return best_tour, best_length, found_time, initial_tour, initial_length, lengths, temps


def generate_points(n, distribution="uniform"):
    """Generate n random points in 2D space with a given distribution."""
    match distribution:
        case "uniform":
            return np.random.rand(n, 2) * 100
        case "normal":
            centers = [(25, 25), (25, 75), (75, 25), (75, 75)]
            points = []
            for _ in range(n):
                center = random.choice(centers)
                points.append(np.random.normal(loc=center, scale=10, size=(1, 2))[0])
            return np.array(points)
        case "clusters":
            clusters = [(10, 10), (90, 90), (10, 90), (90, 10), (50, 50), (25, 75), (75, 25), (25, 25), (75, 75)]
            points = []
            for _ in range(n):
                cluster = random.choice(clusters)
                points.append(np.random.normal(loc=cluster, scale=5, size=(1, 2))[0])
            return np.array(points)


def distance_matrix(points):
    """Compute the distance matrix for given points."""
    n = len(points)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist_matrix[i, j] = np.linalg.norm(points[i] - points[j])
    return dist_matrix


def tour_length(tour, dist_matrix):
    """Calculate the total length of a given tour."""
    return sum(dist_matrix[tour[i], tour[i + 1]] for i in range(len(tour) - 1)) + dist_matrix[tour[-1], tour[0]]


def plot_tour(ax, points, tour, title):
    """Plot the given TSP tour."""
    ordered_points = np.array([points[i] for i in tour] + [points[tour[0]]])
    ax.set_title(title)
    ax.plot(ordered_points[:, 0], ordered_points[:, 1], 'bo-')
    ax.scatter(points[:, 0], points[:, 1], c='red')


def plot_length(ax, lengths, title):
    """Plot the tour lengths over iterations."""
    x_values, y_values = zip(*lengths)
    ax.plot(x_values, y_values, color='green', lw=0.1)
    ax.set_title(title)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Tour length")


def plot_temperature(ax, temps, title):
    """Plot the temperature over iterations."""
    x_values, y_values = zip(*temps)
    ax.plot(x_values, y_values, color='red', lw=1)
    ax.set_title(title)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Temperature")


def plot_results(points, best_tour, best_length, found_time, initial_tour, initial_length, lengths, temps, n, dist):
    """Plot the results of the TSP simulation."""
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"TSP solution for n={n}, distribution: {dist}", fontsize=16)

    plot_tour(axs[0, 0], points, initial_tour, title=f"Before annealing: length = {initial_length:.2f}")
    plot_tour(axs[0, 1], points, best_tour, title=f"After annealing: length = {best_length:.2f}, found in {found_time:.2f}s")
    plot_length(axs[1, 0], lengths, title="Tour length over iterations")
    plot_temperature(axs[1, 1], temps, title="Temperature over iterations")

    plt.show()


def main():
    """Run the TSP simulation with different parameters."""
    distributions = ["uniform", "normal", "clusters"]
    n_values = [20, 50, 100]

    for n in n_values:

        print(f"n={n}")

        for dist in distributions:

            points = generate_points(n, dist)
            start = time.time()
            best_tour, best_length, found_time, initial_tour, initial_length,\
                lengths, temps = simulated_annealing(points)
            elapsed = time.time() - start

            print(f"Distribution: {dist}")
            print(f"Best tour length: {best_length:.2f}, found in {found_time:.2f}s")
            print(f"Total elapsed time: {elapsed:.2f}s")
            print()

            plot_results(points, best_tour, best_length, found_time, 
                         initial_tour, initial_length, lengths, temps, n, dist)

        print("=" * 20)
        print()

if __name__ == "__main__":
    main()
