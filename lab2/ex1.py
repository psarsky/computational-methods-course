import numpy as np
import time


def gauss_jordan(A, b):
    n = len(A)
    Ab = np.hstack([A.astype(float), b.reshape(-1, 1).astype(float)])
    
    for i in range(n):
        max_row = np.argmax(abs(Ab[i:, i])) + i
        if i != max_row:
            Ab[[i, max_row]] = Ab[[max_row, i]]
        
        Ab[i] = Ab[i] / Ab[i, i]
        
        for j in range(n):
            if i != j:
                Ab[j] -= Ab[i] * Ab[j, i]
    
    return Ab[:, -1]


def benchmark():
    sizes = [500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400]
    
    for n in sizes:
        A = np.random.rand(n, n)
        b = np.random.rand(n)

        start = time.time()
        gj_result = gauss_jordan(A, b)
        gj_time = time.time() - start
        
        start = time.time()
        np_result = np.linalg.solve(A, b)
        np_time = time.time() - start

        success = np.allclose(gj_result, np_result) 

        print(f"Size {n}x{n}: Gauss-Jordan = {gj_time:.4f}s, NumPy = {np_time:.4f}s, Ratio = {gj_time / np_time:.4f}, Success = {success}")


benchmark()