"""Test functions for numerical integration methods."""
import numpy as np


def f1(x):
    """Function for testing: exp(-x^2) * (ln(x))^2."""
    return np.exp(-x**2) * (np.log(x))**2

def f2(x):
    """Function for testing: 1 / (x^3 - 2x - 5)."""
    return 1 / (x**3 - 2*x - 5)

def f3(x):
    """Function for testing: x^5 * exp(-x) * sin(x)."""
    return x**5 * np.exp(-x) * np.sin(x)

def f4(x, y):
    """Function for testing: 1 / (sqrt(x + y) * (1 + x + y))."""
    return 1 / (np.sqrt(x + y) * (1 + x + y))

def f5(x, y):
    """Function for testing: x^2 + y^2."""
    return x**2 + y**2
