import numpy as np
import matplotlib.pyplot as plt


# a)


def logistic_map(x, r):
    return r * x * (1 - x)


r_values = np.linspace(1, 4, 1000)
iterations = 1000
last = 200
x0_values = [0.1, 0.3, 0.5, 0.7, 0.9]

plt.figure(figsize=(12, 6))

for x0 in x0_values:
    x_results = []
    for r in r_values:
        x = x0
        print("x0:", x0, "r:", r)
        trajectory = []
        for _ in range(iterations):
            x = logistic_map(x, r)
            if _ >= (iterations - last):
                trajectory.append(x)
        x_results.append(trajectory)

    for i in range(len(r_values)):
        plt.scatter([r_values[i]] * last, x_results[i], s=0.1, label=f'$x_0 = {x0}$' if i == 0 else "", alpha=0.6)

plt.xlabel("r")
plt.ylabel("$x_n$")
plt.title("Diagram bifurkacyjny odwzorowania logistycznego dla różnych wartości $x_0$")
plt.legend()
plt.show()

# Interpretation:
# r < 3: x_n stabilizes to a fixed value;
# 3 < r < 3.5: x_n oscillates between two values;
# r > 3.5: number of oscillations increases rapidly, leading to chaos;
# r = 4: full chaos - no predictable pattern.


# b)


def logistic_map_single(x, r):
    return np.float32(r * x * (1 - x))


def logistic_map_double(x, r):
    return r * x * (1 - x)


r_values = [3.75, 3.78, 3.8]
x0 = 0.4
iterations = 100

plt.figure(figsize=(12, 6))

for r in r_values:
    x_single = np.float32(x0)
    x_double = float(x0)
    
    x_vals_single = []
    x_vals_double = []
    
    for _ in range(iterations):
        x_single = logistic_map_single(x_single, np.float32(r))
        x_double = logistic_map_double(x_double, r)
        x_vals_single.append(x_single)
        x_vals_double.append(x_double)

    plt.plot(range(iterations), x_vals_single, linestyle='dashed', label=f'Pojedyncza precyzja, r={r}')
    plt.plot(range(iterations), x_vals_double, linestyle='solid', label=f'Podwójna precyzja, r={r}')

plt.xlabel("Iteracje")
plt.ylabel("$x_n$")
plt.title("Porównanie trajektorii dla pojedynczej i podwójnej precyzji")
plt.legend()
plt.show()