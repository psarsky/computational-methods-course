"""Excercise 1: Simulated Annealing for TSP"""
import math
import random

import matplotlib.pyplot as plt
import numpy as np


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
    return sum(dist_matrix[tour[i], tour[i+1]] for i in range(len(tour) - 1)) + dist_matrix[tour[-1], tour[0]]


def simulated_annealing(points, max_iter=1000, initial_temp=1000, cooling_rate=0.995, swap_type="consecutive"):
    """Solve the TSP using simulated annealing."""
    n = len(points)
    dist_matrix = distance_matrix(points)
    current_tour = list(range(n))
    random.shuffle(current_tour)
    current_length = tour_length(current_tour, dist_matrix)
    best_tour = current_tour.copy()
    best_length = current_length
    temperature = initial_temp

    for _ in range(max_iter):
        if swap_type == "consecutive":
            i = random.randint(0, n-2)
            j = i + 1
        else:
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

        temperature *= cooling_rate

    return best_tour, best_length


def plot_tour(points, tour, title="TSP Solution"):
    """Plot the best found TSP tour."""
    plt.figure(figsize=(8, 8))
    ordered_points = np.array([points[i] for i in tour] + [points[tour[0]]])
    plt.plot(ordered_points[:, 0], ordered_points[:, 1], 'bo-')
    plt.scatter(points[:, 0], points[:, 1], c='red')
    plt.title(title)
    plt.show()


def main():
    """Run the TSP simulation with different parameters."""
    distributions = ["uniform", "normal", "clusters"]
    n_values = [20, 50, 100]
    
    for n in n_values:
        for dist in distributions:
            points = generate_points(n, dist)
            best_tour, best_length = simulated_annealing(points)
            plot_tour(points, best_tour, title=f"n={n}, dist={dist}")

if __name__ == "__main__":
    main()
