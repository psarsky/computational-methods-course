# Bisection method

# Stop conditions:
# 1. iteration limit
# 2. |b - a| < tol
# 3. f(c) < precision


import mpmath


def bisection_method(f, a, b, precision, tol, iter_limit):
    # Convert a and b to given precision
    mpmath.mp.dps = precision
    a, b = mpmath.mpf(a), mpmath.mpf(b)
    
    # Initial condition
    if mpmath.sign(f(a)) == mpmath.sign(f(b)):
        raise ValueError("Function must have opposite signs at a and b.")
    
    iterations = 0
    
    # Stop condition 1. and 2.
    while abs(b - a) > tol and iterations < iter_limit:
        # Numerically stable mean
        c = a + (b - a) / 2

        # Stop condition 3.
        if f(c) == 0:
            return c, iterations
        
        if mpmath.sign(f(a)) == mpmath.sign(f(c)):
            a = c
        else:
            b = c
        
        iterations += 1
    
    return a + (b - a) / 2, iterations
