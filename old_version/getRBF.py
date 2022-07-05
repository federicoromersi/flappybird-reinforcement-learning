import numpy as np
from build_tiles import build_tiles
import math

def getRBF(s, a, sigma, gridx, gridy, M, N, A):
    
    # numero di celle
    nCells = pow(M, 2)    # sistemare numero celle
    d = A*N*nCells    # dimensione spazio di stato lineare

    # spazio di stato lineare
    state = np.zeros((d, 1))


    for i1 in range(0, N):
        for i2 in range(0, M):
            for i3 in range(0, M):
                ind = np.ravel_multi_index((i3, i2, i1, a), dims=(M, M, N, A), order='C')
                state[ind] = math.exp(-1/pow(sigma,1) * pow(np.linalg.norm(np.array([s[0], s[1]]) - np.array([gridx[i1,i2], gridy[i1,i3]])), 2))

    
    return state