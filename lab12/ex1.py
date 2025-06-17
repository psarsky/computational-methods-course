import matplotlib.pyplot as plt
import numpy as np


def speed_function(t):
    if 0 <= t <= 8:
        return 15 + 4 * t
    if 8 < t <= 12:
        return 47 + 2 * (t - 8)
    if 12 < t <= 18:
        return 55 - 1.5 * (t - 12)
    if 18 < t <= 25:
        return 46 - 3 * (t - 18)
    if 25 < t <= 30:
        return 25 - 2 * (t - 25)
    if 30 < t <= 35:
        return 15 - 1.5 * (t - 30)
    if 35 < t <= 40:
        return 7.5 - 1 * (t - 35)
    return max(0, 2.5 - 0.5 * (t - 40))


def trapezoid_rule(f, a, b, n):
    h = (b - a) / n
    result = 0.5 * (f(a) + f(b))
    for i in range(1, n):
        result += f(a + i * h)
    return result * h


def main():
    t_values = np.linspace(0, 40, 1000)
    speed_values = [speed_function(t) for t in t_values]

    total_distance = trapezoid_rule(speed_function, 0, 40, 1000) / 3600

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    plt.plot(t_values, speed_values, 'b-', linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Speed (km/h)')
    plt.title('Speed vs time')
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 1, 2)
    distances = []
    time_points = []
    for t in np.linspace(0, 40, 300):
        if t > 0:
            dist = trapezoid_rule(speed_function, 0, t, 100) / 3600
            distances.append(dist)
            time_points.append(t)

    plt.plot(time_points, distances, 'r-', linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Distance traveled (km)')
    plt.title('Distance traveled vs time')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print(f"Total traveled distance: {total_distance:.4f} km")


if __name__ == "__main__":
    main()
