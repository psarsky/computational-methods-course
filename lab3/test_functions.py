import mpmath


def f_1(x):
    return mpmath.cos(x) * mpmath.cosh(x) - 1


def f_2(x):
    return 1 / x - mpmath.tan(x)


def f_3(x):
    return 2 ** (-x) + mpmath.exp(x) + 2 * mpmath.cos(x) - 6


def df_1(x):
    return mpmath.cos(x) * mpmath.sinh(x) - mpmath.sin(x) * mpmath.cosh(x)


def df_2(x):
    return -1 / (x ** 2) - mpmath.sec(x) ** 2


def df_3(x):
    return -2 ** (-x) * mpmath.log(2) + mpmath.exp(x) - 2 * mpmath.sin(x)
