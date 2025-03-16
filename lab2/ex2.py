import numpy as np


def lu_in_place(A):
    n = A.shape[0]
    
    for i in range(n):
        for j in range(i+1, n):
            A[j, i] /= A[i, i]
            A[j, i+1:] -= A[j, i] * A[i, i+1:]
    
    return A


def deconstruct_LU(A):
    n = A.shape[0]
    L = np.tril(A, k=-1) + np.eye(n)
    U = np.triu(A)
    return L, U


def test_lu_factorization():
    sizes = [100, 500, 1000, 2000]
    
    for n in sizes:
        A = np.random.rand(n, n)
        A_orig = A.copy()
        
        lu_in_place(A)
        L, U = deconstruct_LU(A)
        
        error = np.linalg.norm(A_orig - L @ U)
        print(f"Size {n}x{n}: Factorization error = {error:.6e}")


test_lu_factorization()
