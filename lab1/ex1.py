import numpy as np
import matplotlib.pyplot as plt
import time


# 1
N = 10**7
v = np.float32(0.1)
array = np.full(N, v)

computed_sum_lp = np.float32(0.0)
for i in range(N):
    computed_sum_lp += array[i]

computed_sum_py = sum(array)

computed_sum_np = np.sum(array)

exact_sum = N * v


# 2
absolute_error_lp = abs(computed_sum_lp - exact_sum)
absolute_error_py = abs(computed_sum_py - exact_sum)
absolute_error_np = abs(computed_sum_np - exact_sum)
relative_error_lp = absolute_error_lp / exact_sum
relative_error_py = absolute_error_py / exact_sum
relative_error_np = absolute_error_np / exact_sum
print(f'Computed sum (lp): {computed_sum_lp}')
print(f'Computed sum (py): {computed_sum_py}')
print(f'Computed sum (np): {computed_sum_np}')
print(f'Exact sum: {exact_sum}')
print(f'Absolute error (lp): {absolute_error_lp}')
print(f'Absolute error (py): {absolute_error_py}')
print(f'Absolute error (np): {absolute_error_np}')
print(f'Relative error (lp): {relative_error_lp}')
print(f'Relative error (py): {relative_error_py}')
print(f'Relative error (np): {relative_error_np}')


# 3
errors = []
current_sum = np.float32(0.0)
for i in range(0, N):
    current_sum += array[i]
    if (i+1) % 25000 == 0:
        exact_partial_sum = (i+1) * v
        abs_error = abs(current_sum - exact_partial_sum)
        rel_error = abs_error / exact_partial_sum
        errors.append(rel_error)

plt.plot(range(25000, N+1, 25000), errors)
plt.xlabel('No. of added elements')
plt.ylabel('Relative error')
plt.title('Relative error growth')
plt.show()


# 4
def recursive_sum(arr):
    if len(arr) == 1:
        return arr[0]
    mid = len(arr) // 2
    return recursive_sum(arr[:mid]) + recursive_sum(arr[mid:])

recursive_result = recursive_sum(array)
print(f'Recursive sum: {recursive_result}')


# 5
recursive_absolute_error = abs(recursive_result - exact_sum)
recursive_relative_error = recursive_absolute_error / exact_sum
print(f'Recursive Absolute Error: {recursive_absolute_error}')
print(f'Recursive Relative Error: {recursive_relative_error}')


# 6
start_time = time.time()
sum_py = sum(array)
time_py = time.time() - start_time

start_time = time.time()
sum_np = np.sum(array)
time_np = time.time() - start_time

start_time = time.time()
sum_rec = recursive_sum(array)
time_rec = time.time() - start_time

print(f'Python sum time: {time_py:.5f} seconds')
print(f'Numpy sum time: {time_np:.5f} seconds')
print(f'Recursive sum time: {time_rec:.5f} seconds')


# 7
v_1 = np.float32(0.5315)
error_prone_array = np.full(N, v_1, dtype=np.float32)
recursive_error_test = recursive_sum(error_prone_array)
exact_error_test = N * v_1

print(f'Recursive sum (error-prone input): {recursive_error_test}')
print(f'Exact sum (error-prone input): {exact_error_test}')