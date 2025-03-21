import numpy as np


def partial_sum_zeta_float32(s, n):
    sum_ = np.float32(0.0)

    for k in range(1, n+1):
        val = np.float32(1.0 / (k**s))
        sum_ += val

    return sum_


def partial_sum_zeta_float64(s, n):
    sum_ = 0.0

    for k in range(1, n+1):
        val = 1.0 / (k**s)
        sum_ += val

    return sum_


def partial_sum_eta_float32(s, n):
    sum_ = np.float32(0.0)
    sign = np.float32(1.0)

    for k in range(1, n+1):
        val = np.float32(sign * (1.0/(k**s)))
        sum_ += val
        sign = -sign

    return sum_


def partial_sum_eta_float64(s, n):
    sum_ = 0.0
    sign = 1.0

    for k in range(1, n+1):
        val = sign * (1.0/(k**s))
        sum_ += val
        sign = -sign

    return sum_


def partial_sum_zeta_float32_rev(s, n):
    sum_ = np.float32(0.0)

    for k in range(n, 0, -1):
        val = np.float32(1.0 / (k**s))
        sum_ += val

    return sum_


def partial_sum_zeta_float64_rev(s, n):
    sum_ = 0.0

    for k in range(n, 0, -1):
        val = 1.0 / (k**s)
        sum_ += val

    return sum_


def partial_sum_eta_float32_rev(s, n):
    sum_ = np.float32(0.0)
    sign = np.float32((-1)**(n-1))

    for k in range(n, 0, -1):
        val = np.float32(sign * (1.0 / (k**s)))
        sum_ += val
        sign = -sign

    return sum_


def partial_sum_eta_float64_rev(s, n):
    sum_ = 0.0
    sign = (-1)**(n-1)

    for k in range(n, 0, -1):
        val = sign * (1.0 / (k**s))
        sum_ += val
        sign = -sign

    return sum_


s_values = [2, 3.6667, 7, 10]
n_values = [50, 100, 200, 500, 1000]

for s in s_values:
    print(f'\n================ s = {s} ================')

    for n in n_values:
        z32_fwd = partial_sum_zeta_float32(s, n)
        z32_bwd = partial_sum_zeta_float32_rev(s, n)
        z64_fwd = partial_sum_zeta_float64(s, n)
        z64_bwd = partial_sum_zeta_float64_rev(s, n)

        e32_fwd = partial_sum_eta_float32(s, n)
        e32_bwd = partial_sum_eta_float32_rev(s, n)
        e64_fwd = partial_sum_eta_float64(s, n)
        e64_bwd = partial_sum_eta_float64_rev(s, n)

        print(f'n = {n}')
        print(f'  Zeta float32 forward:  {z32_fwd:.16f}')
        print(f'  Zeta float32 backward: {z32_bwd:.16f}')
        print(f'  Zeta float64 forward:  {z64_fwd:.16f}')
        print(f'  Zeta float64 backward: {z64_bwd:.16f}')
        print(f'  Eta  float32 forward:  {e32_fwd:.16f}')
        print(f'  Eta  float32 backward: {e32_bwd:.16f}')
        print(f'  Eta  float64 forward:  {e64_fwd:.16f}')
        print(f'  Eta  float64 backward: {e64_bwd:.16f}')

# Forward vs backward
# Rounding error can be reduced by summing the elements form the smallest to
# the largest values - by starting with the smallest values the partial sums
# don't lose as much precision thanks to their small size.
