"""Excercise 1: Simulated annealing for TSP"""
import copy
import math
import random
import time

import matplotlib.pyplot as plt
import numpy as np


def simulated_annealing(
        points,
        temperature=100,
        temp_function=lambda x, _: x*0.995,
        threshold=1e-2,
        swap_type="arbitrary"
    ):
    """Solve TSP using simulated annealing."""
    point_amount = len(points)
    dist_matrix = distance_matrix(points)

    current_path = list(range(point_amount))
    random.shuffle(current_path)
    current_length = path_length(current_path, dist_matrix)

    initial_path = copy.deepcopy(current_path)
    initial_length = current_length

    best_path = copy.deepcopy(current_path)
    best_length = current_length
    found_time = t_0 = time.perf_counter()

    lengths = [(0, initial_length)]
    temps = [(0, temperature)]
    paths = [copy.deepcopy(initial_path)]

    while temperature > threshold:
        for _ in range(100):
            match swap_type:
                case "consecutive":
                    i = random.randint(0, point_amount - 2)
                    j = i + 1
                case "arbitrary":
                    i, j = random.sample(range(point_amount), 2)

            current_path[i], current_path[j] = current_path[j], current_path[i]
            new_length = path_length(current_path, dist_matrix)

            if new_length < current_length or random.random() < math.exp((current_length - new_length) / temperature):
                current_length = new_length

                if new_length < best_length:
                    best_path = copy.deepcopy(current_path)
                    best_length = new_length
                    found_time = time.perf_counter() - t_0

            else:
                current_path[i], current_path[j] = current_path[j], current_path[i]

            paths.append(copy.deepcopy(current_path))

            lengths.append((time.perf_counter() - t_0, current_length))

        timestamp = time.perf_counter() - t_0
        temperature = temp_function(temperature, timestamp)
        temps.append((timestamp, temperature))

    return best_path, best_length, found_time, initial_path, initial_length, lengths, temps, paths


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
                points.append(np.random.normal(loc=center, scale=5, size=(1, 2))[0])
            return np.array(points)
        case "clusters":
            clusters = [(10, 10), (90, 90), (10, 90), (90, 10), (50, 50), (25, 75), (75, 25), (25, 25), (75, 75)]
            points = []
            for _ in range(n):
                cluster = random.choice(clusters)
                points.append(np.random.normal(loc=cluster, scale=2, size=(1, 2))[0])
            return np.array(points)


def distance_matrix(points):
    """Compute the distance matrix for given points."""
    point_amount = len(points)
    dist_matrix = np.zeros((point_amount, point_amount))
    for i in range(point_amount):
        for j in range(point_amount):
            dist_matrix[i, j] = np.linalg.norm(points[i] - points[j])
    return dist_matrix


def path_length(path, dist_matrix):
    """Calculate the total length of a given path."""
    return sum(dist_matrix[path[i], path[i + 1]] for i in range(len(path) - 1)) + dist_matrix[path[-1], path[0]]


def plot_path(ax, points, path, title):
    """Plot the given TSP path."""
    ordered_points = np.array([points[i] for i in path] + [points[path[0]]])
    ax.set_title(title)
    ax.plot(ordered_points[:, 0], ordered_points[:, 1], 'bo-')
    ax.scatter(points[:, 0], points[:, 1], c='red')


def plot_length(ax, lengths, title):
    """Plot the path lengths over iterations."""
    x_values, y_values = zip(*lengths)
    ax.plot(x_values, y_values, color='green', lw=0.5)
    ax.set_title(title)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Path length")


def plot_temperature(ax, temps, title):
    """Plot the temperature over iterations."""
    x_values, y_values = zip(*temps)
    ax.plot(x_values, y_values, color='red', lw=1)
    ax.set_title(title)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Temperature")


def plot_results(
        points,
        best_path,
        best_length,
        found_time,
        initial_path,
        initial_length,
        lengths, temps,
        n,
        dist,
        swap_type
    ):
    """Plot the results of the TSP simulation."""
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"TSP solution for n={n}, distribution: {dist}, swap type: {swap_type}", fontsize=16)

    plot_path(axs[0, 0], points, initial_path, title=f"Initial path: length = {initial_length:.2f}")
    plot_path(axs[0, 1], points, best_path,
              title=f"After annealing: length = {best_length:.2f}, found in {found_time:.2f}s")
    plot_length(axs[1, 0], lengths, title="Path length over iterations")
    plot_temperature(axs[1, 1], temps, title="Temperature over iterations")

    plt.show()


def main():
    """Run the TSP simulation with different parameters."""
    n_values = [20, 50, 100]
    distributions = ["uniform", "normal", "clusters"]
    swap_types = ["arbitrary", "consecutive"]

    for n in n_values:
        print(f"---------- n = {n} ----------\n")
        for dist in distributions:
            points = generate_points(n, dist)
            results = []
            for swap_type in swap_types:
                start = time.time()
                best_path, best_length, found_time, initial_path, initial_length, lengths, temps, paths =\
                    simulated_annealing(points, swap_type=swap_type)
                elapsed = time.time() - start

                results.append((swap_type, best_length))

                print(f"Distribution: {dist}, swap type: {swap_type}")
                print(f"Initial path length: {initial_length:.2f}")
                print(f"Best path length: {best_length:.2f}, found in {found_time:.2f}s")
                print(f"Total elapsed time: {elapsed:.2f}s\n")

                plot_results(
                    points,
                    best_path,
                    best_length,
                    found_time,
                    initial_path,
                    initial_length,
                    lengths,
                    temps,
                    n,
                    dist,
                    swap_type
                )

            print("Shortest path comparison for different swap types:")
            for swap_type, best_length in results:
                print(f"{swap_type}: {best_length:.2f}")
            print()
            if dist != "clusters":
                print("--------------------\n")

        print("=" * 20)
        print()

if __name__ == "__main__":
    main()
