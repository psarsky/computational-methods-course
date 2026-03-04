"""Excercise 1: Simulated annealing for TSP"""
import copy
import math
import random
import time

from matplotlib import rcParams
from util import distance_matrix, generate_points, path_length
from vis import animate, plot_results


rcParams['animation.embed_limit'] = 200


def simulated_annealing(
        points, temperature=100, temp_function=lambda x, _: x*0.995, threshold=1e-2, swap_type="arbitrary"
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

                animate(f"n{n}_{dist}_{swap_type}.html", paths, points, best_path, found_time/elapsed)

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


if __name__ == "__main__":
    main()
