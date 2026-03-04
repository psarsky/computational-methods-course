"""Exercise 2: Kahan's algorithm."""
import time

import numpy as np


def kahan(arr):
    """Kahan's algorithm for summation."""
    res = np.float32(0.0)
    error = np.float32(0.0)

    for _, element in enumerate(arr):
        y = element - error
        temp = res + y
        error = (temp - res) - y
        res = temp

    return res


def recursive_sum(arr):
    """Recursively computes the sum of an array."""
    if len(arr) == 1:
        return arr[0]

    mid = len(arr) // 2

    return recursive_sum(arr[:mid]) + recursive_sum(arr[mid:])


# 1
N = 10**7
v = np.float32(0.1)
array = np.full(N, v)

computed_sum_py = sum(array)
computed_sum_np = np.sum(array)
computed_sum_rc = recursive_sum(array)
computed_sum_kh = kahan(array)

exact_sum = N * v

absolute_error_py = abs(computed_sum_py - exact_sum)
absolute_error_np = abs(computed_sum_np - exact_sum)
absolute_error_rc = abs(computed_sum_rc - exact_sum)
absolute_error_kh = abs(computed_sum_kh - exact_sum)
relative_error_py = absolute_error_py / exact_sum
relative_error_np = absolute_error_np / exact_sum
relative_error_rc = absolute_error_rc / exact_sum
relative_error_kh = absolute_error_kh / exact_sum
print(f'Computed sum (py): {computed_sum_py}')
print(f'Computed sum (np): {computed_sum_np}')
print(f'Computed sum (rc): {computed_sum_rc}')
print(f'Computed sum (kh): {computed_sum_kh}')
print(f'Exact sum: {exact_sum}')
print(f'Absolute error (py): {absolute_error_py}')
print(f'Absolute error (np): {absolute_error_np}')
print(f'Absolute error (rc): {absolute_error_rc}')
print(f'Absolute error (kh): {absolute_error_kh}')
print(f'Relative error (py): {relative_error_py}')
print(f'Relative error (np): {relative_error_np}')
print(f'Relative error (rc): {relative_error_rc}')
print(f'Relative error (kh): {relative_error_kh}')


# 2
# Kahan's algorithm has better computational properties thanks to reducing the
# accumulated rounding error. The "err" variable is utilized to compensate
# the errors by storing values lost while adding floating point numbers.


# 3
start_time = time.time()
sum_py = sum(array)
time_py = time.time() - start_time

start_time = time.time()
sum_np = np.sum(array)
time_np = time.time() - start_time

start_time = time.time()
sum_rec = recursive_sum(array)
time_rec = time.time() - start_time

start_time = time.time()
sum_kh = kahan(array)
time_kh = time.time() - start_time

print(f'Python sum time: {time_py:.5f} seconds')
print(f'Numpy sum time: {time_np:.5f} seconds')
print(f'Recursive sum time: {time_rec:.5f} seconds')
print(f'Kahan sum time: {time_kh:.5f} seconds')
