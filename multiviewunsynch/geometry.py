# Useful geometrical functions

import numpy as np

def X_(v):
    """Creates a 3x3 skew symmetric matrix from the input vector  

    A skew symmetric matrix has the property A^T = -A
    3x3 skew symmetric matrices represent cross products as matrix multiplications such that x_(a)*b = a x b
    """
    M = np.zeros((3,3))
    M[0,1] = -v[2]
    M[0,2] = v[1]
    M[1,0] = v[2]
    M[1,2] = -v[0]
    M[2,0] = -v[1]
    M[2,1] = v[2]
    return M