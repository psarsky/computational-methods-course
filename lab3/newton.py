# Metoda stycznych
# Newtona
# Newtona-Raphsona

# dodatkowo - x^x (funkcja specjalna Lamberta)


import mpmath


def newton_method(f, df, x0, precision, tol, iter_limit):
    mpmath.mp.dps = precision
    x = mpmath.mpf(x0)
    
    iterations = 0

    while iterations < iter_limit:
        fx = f(x)
        dfx = df(x)
        
        x_new = x - fx / dfx
        
        if abs(x_new - x) < tol:
            return x_new, iterations
        
        x = x_new
        iterations += 1
    
    return x, iterations