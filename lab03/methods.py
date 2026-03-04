"""Methods for finding roots of a nonlinear function."""
import mpmath


def bisection(f, a, b, precision, tol, iter_limit):
    """Bisection method"""
    mpmath.mp.dps = precision
    a, b = mpmath.mpf(a), mpmath.mpf(b)

    if mpmath.sign(f(a)) == mpmath.sign(f(b)):
        raise ValueError("Function must have opposite signs at a and b.")

    iterations = 0

    while abs(b - a) > tol and iterations < iter_limit:
        c = a + (b - a) / 2

        if f(c) == 0:
            return c, iterations

        if mpmath.sign(f(a)) == mpmath.sign(f(c)):
            a = c
        else:
            b = c

        iterations += 1

    return a + (b - a) / 2, iterations


def newton(f, df, x0, precision, tol, iter_limit):
    """Newton's method"""
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


def secant(f, x0, x1, precision, tol, iter_limit):
    """Secant method"""
    mpmath.mp.dps = precision
    x0, x1 = mpmath.mpf(x0), mpmath.mpf(x1)

    iterations = 0

    while iterations < iter_limit:
        f_x0, f_x1 = f(x0), f(x1)
        if f_x1 - f_x0 == 0:
            raise ValueError("Zero division error in secant method.")

        x_new = x1 - f_x1 * (x1 - x0) / (f_x1 - f_x0)
        if abs(x_new - x1) < tol:
            return x_new, iterations

        x0, x1 = x1, x_new
        iterations += 1

    return x1, iterations


def combo(f, df, a, b, precision, tol, iter_limit):
    """Bisection + Newton's method combination"""
    mpmath.mp.dps = precision
    a, b = mpmath.mpf(a), mpmath.mpf(b)

    if mpmath.sign(f(a)) == mpmath.sign(f(b)):
        raise ValueError("Function must have opposite signs at a and b.")

    iterations = 0

    while abs(b - a) > tol and iterations < 5:
        c = a + (b - a) / 2

        if f(c) == 0:
            return c, iterations

        if mpmath.sign(f(a)) == mpmath.sign(f(c)):
            a = c
        else:
            b = c

        iterations += 1

    x = a + (b - a) / 2

    while iterations < iter_limit:
        fx = f(x)
        dfx = df(x)

        x_new = x - fx / dfx

        if abs(x_new - x) < tol or f(x_new) == 0:
            return x_new, iterations

        x = x_new
        iterations += 1

    return x, iterations


def bisection_vis(f, a, b, precision, tol, iter_limit):
    """Bisection method - visualization"""
    mpmath.mp.dps = precision
    a, b = mpmath.mpf(a), mpmath.mpf(b)

    if mpmath.sign(f(a)) == mpmath.sign(f(b)):
        raise ValueError("Function must have opposite signs at a and b.")

    iterations = 0
    steps = []

    while abs(b - a) > tol and iterations < iter_limit:
        c = a + (b - a) / 2
        steps.append(float(c))

        if f(c) == 0:
            return steps

        if mpmath.sign(f(a)) == mpmath.sign(f(c)):
            a = c
        else:
            b = c

        iterations += 1

    return steps


def newton_vis(f, df, x0, precision, tol, iter_limit):
    """Newton's method - visualization"""
    mpmath.mp.dps = precision
    x = mpmath.mpf(x0)

    iterations = 0
    steps = [float(x)]

    while iterations < iter_limit:
        fx = f(x)
        dfx = df(x)

        x_new = x - fx / dfx
        steps.append(float(x_new))

        if abs(x_new - x) < tol or f(x_new) == 0:
            return steps

        x = x_new
        iterations += 1

    return steps


def secant_vis(f, x0, x1, precision, tol, iter_limit):
    """Secant method - visualization"""
    mpmath.mp.dps = precision
    x0, x1 = mpmath.mpf(x0), mpmath.mpf(x1)

    iterations = 0
    steps = [float(x0), float(x1)]

    while iterations < iter_limit:
        f_x0, f_x1 = f(x0), f(x1)
        if f_x1 - f_x0 == 0:
            raise ValueError("Zero division error in secant method.")

        x_new = x1 - f_x1 * (x1 - x0) / (f_x1 - f_x0)
        steps.append(float(x_new))
        if abs(x_new - x1) < tol:
            return steps

        x0, x1 = x1, x_new
        iterations += 1

    return steps


def combo_vis(f, df, a, b, precision, tol, iter_limit):
    """Bisection + Newton's method combination - visualization"""
    mpmath.mp.dps = precision
    a, b = mpmath.mpf(a), mpmath.mpf(b)

    if mpmath.sign(f(a)) == mpmath.sign(f(b)):
        raise ValueError("Function must have opposite signs at a and b.")

    iterations = 0
    steps = []

    while abs(b - a) > tol and iterations < 5:
        c = a + (b - a) / 2
        steps.append(float(c))

        if f(c) == 0:
            return steps

        if mpmath.sign(f(a)) == mpmath.sign(f(c)):
            a = c
        else:
            b = c

        iterations += 1

    x = a + (b - a) / 2

    while iterations < iter_limit:
        fx = f(x)
        dfx = df(x)

        x_new = x - fx / dfx
        steps.append(float(x_new))

        if abs(x_new - x) < tol or f(x_new) == 0:
            return steps

        x = x_new
        iterations += 1

    return steps
