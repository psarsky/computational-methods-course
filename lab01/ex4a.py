"""Exercise 4a: Bifurcation diagram for logistic map with different initial conditions."""
import matplotlib.pyplot as plt
import numpy as np


def logistic_map(x, r):
    """Compute the logistic map."""
    return r * x * (1 - x)


r_values = np.linspace(1, 4, 1000)
iterations = 1000
last = 200
x0_values = [0.1, 0.3, 0.5, 0.7, 0.9]

plt.figure(figsize=(12, 6))

colors = ['red', 'orange', 'yellow', 'green', 'blue']

for x0 in x0_values:
    x_results = []
    for r_val in r_values:
        x_val = x0
        trajectory = []
        for _ in range(iterations):
            x_val = logistic_map(x_val, r_val)
            if _ >= (iterations - last):
                trajectory.append(x_val)
        x_results.append(trajectory)

    for i, r_val in enumerate(r_values):
        plt.scatter([r_val] * last, x_results[i], s=0.1, label=f'$x_0 = {x0}$' if i == 0 else "",
                    alpha=0.6, color=colors[x0_values.index(x0)])

plt.xlabel("r")
plt.ylabel("$x_n$")
plt.title("Bifurcation diagram for different $x_0$ values")
plt.legend()
plt.show()

# Interpretation:
# r < 3: x_n stabilizes to a fixed value;
# 3 < r < 3.5: x_n oscillates between two values;
# r > 3.5: number of oscillations increases rapidly, leading to chaos;
# r = 4: full chaos - no predictable pattern.
