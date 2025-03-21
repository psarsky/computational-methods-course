import mpmath


# Bisection method

# Stop conditions:
# 1. iteration limit
# 2. |b - a| < tol
# 3. f(c) < precision
def bisection(f, a, b, precision, tol, iter_limit):
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

# Newton method
def newton(f, df, x0, precision, tol, iter_limit):
    mpmath.mp.dps = precision
    x = mpmath.mpf(x0)
    
    iterations = 0

    while iterations < iter_limit:
        fx = f(x)
        dfx = df(x)
        
        x_new = x - fx / dfx
        
        if abs(x_new - x) < tol or f(x_new) == 0:
            return x_new, iterations
        
        x = x_new
        iterations += 1
    
    return x, iterations
