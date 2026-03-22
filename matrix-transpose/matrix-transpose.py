import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    # Write code here
    A = np.array(A)
    B = np.zeros((A.shape[1],A.shape[0]))
    for i in range(B.shape[0]):
        for j in range(B.shape[1]):
            B[i,j] = A[j,i]
    return B
