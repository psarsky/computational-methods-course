"""Exercise 4c: Iterations to reach zero for r=4"""
import numpy as np
import matplotlib.pyplot as plt


def count_iterations_to_zero(x0, iter_limit, r=4, dtype=np.float32):
    """Count the number of iterations needed for the logistic map to reach zero."""
    x = dtype(x0)
    for i in range(iter_limit):
        x = dtype(r * x * (1 - x))
        if x == 0:
            print(f"x0 = {x0} converged to 0 after {i} iterations.")
            return i
    return iter_limit


x_values = np.linspace(0.001, 0.999, 100)
iter_lim = int(input("Enter the iteration limit: "))
iterations_needed = [count_iterations_to_zero(x, iter_lim) for x in x_values]

plt.figure(figsize=(10, 5))
plt.plot(x_values, iterations_needed, marker="o", linestyle="")
plt.xlabel("$x_0$")
plt.ylabel("Iterations")
plt.title("Iterations to reach zero for r=4")
plt.show()

# Some x0 values (15/100) need a relatively small amount of iterations (<=2200) to converge to 0.
# For the rest of the values the system remains in chaos for over 1000000 iterations.
