
import numpy as np
from scipy.sparse import csr_matrix

arr = np.array([
    [1,0,0,1,0,0],
    [0,0,2,0,0,1],
    [0,0,0,2,0,0]
    ])

print(f"arr is {arr}")

S = csr_matrix(arr)
print(f"CSR matrix is {S}")

B = S.todense()
print(f"dense matrix is {B}")


