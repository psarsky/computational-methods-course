"""Calculate the speed of a vehicle over time using the trapezoidal rule and plot the results."""

import matplotlib.pyplot as plt
import numpy as np


def speed_function(t):
    """Speed function based on time intervals."""
    intervals = [
        (0, 8, lambda t: 15 + 4 * t),
        (8, 12, lambda t: 47 + 2 * (t - 8)),
        (12, 18, lambda t: 55 - 1.5 * (t - 12)),
        (18, 25, lambda t: 46 - 3 * (t - 18)),
        (25, 30, lambda t: 25 - 2 * (t - 25)),
        (30, 35, lambda t: 15 - 1.5 * (t - 30)),
        (35, 40, lambda t: 7.5 - 1 * (t - 35)),
    ]

    for start, end, speed_calc in intervals:
        if start <= t <= end:
            return speed_calc(t)

    return max(0, 2.5 - 0.5 * (t - 40))


def trapezoid_rule(f, a, b, n):
    """Trapezoidal rule for numerical integration."""
    h = (b - a) / n
    result = 0.5 * (f(a) + f(b))
    for i in range(1, n):
        result += f(a + i * h)
    return result * h


def main():
    """Main function to plot speed and distance traveled over time."""
    t_values = np.linspace(0, 40, 1000)
    speed_values = [speed_function(t) for t in t_values]

    total_distance = trapezoid_rule(speed_function, 0, 40, 1000) / 3600

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    plt.plot(t_values, speed_values, "b-", linewidth=2)
    plt.xlabel("Time (s)")
    plt.ylabel("Speed (km/h)")
    plt.title("Speed vs time")
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 1, 2)
    distances = []
    time_points = []
    for t in np.linspace(0, 40, 300):
        if t > 0:
            dist = trapezoid_rule(speed_function, 0, t, 100) / 3600
            distances.append(dist)
            time_points.append(t)

    plt.plot(time_points, distances, "r-", linewidth=2)
    plt.xlabel("Time (s)")
    plt.ylabel("Distance traveled (km)")
    plt.title("Distance traveled vs time")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print(f"Total traveled distance: {total_distance:.4f} km")


if __name__ == "__main__":
    main()
