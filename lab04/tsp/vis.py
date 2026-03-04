"""Visualization utilities for the TSP problem."""
import copy
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from util import distance_matrix, path_length


def plot_path(ax, points, path, title):
    """Plot the given TSP path."""
    ordered_points = np.array([points[i] for i in path] + [points[path[0]]])
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
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
        points, best_path, best_length, found_time, initial_path, initial_length, lengths, temps, n, dist, swap_type
    ):
    """Plot the results of the TSP simulation."""
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"TSP solution for n={n}, distribution: {dist}, swap type: {swap_type}", fontsize=16)

    plot_path(axs[0, 0], points, initial_path, title=f"Initial path: length = {initial_length:.2f}")
    plot_path(axs[0, 1], points, best_path,
              title=f"After annealing: length = {best_length:.2f}, found in {found_time:.2f}s")
    plot_length(axs[1, 0], lengths, title="Path length over time")
    plot_temperature(axs[1, 1], temps, title="Temperature over time")

    plt.show()


def animate(filename, paths, points, best, pos, framerate=30):
    """Create an animation of the TSP solution."""
    print("Creating animation...")

    best_path = None
    best_length = float('inf')

    def update(frame):
        nonlocal best_path, best_length

        dist_matrix = distance_matrix(points)
        current_length = path_length(frame, dist_matrix)

        if current_length < best_length:
            best_length = current_length
            best_path = copy.deepcopy(frame)

        ax.clear()
        ax.set_title(f"Current length: {current_length:.2f}, Best length: {best_length:.2f}")
        ax.set_xticks([])
        ax.set_yticks([])

        if best_path:
            short_x = [points[x][0] for x in best_path] + [points[best_path[0]][0]]
            short_y = [points[x][1] for x in best_path] + [points[best_path[0]][1]]
            ax.plot(short_x, short_y, color='green', linewidth=5, alpha=0.5, label="Best path")

        x_values = [points[x][0] for x in frame] + [points[frame[0]][0]]
        y_values = [points[x][1] for x in frame] + [points[frame[0]][1]]

        ax.scatter(x_values, y_values, color='red')
        ax.plot(x_values, y_values, color='blue', linewidth=0.5, label="Current path")
        ax.legend()

    fig, ax = plt.subplots(figsize=(8, 8))
    step = len(paths) // 200
    fr = paths[::step]
    fr[int(pos * 200)] = copy.deepcopy(best)    # to make sure the best path is included in the animation sample

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
