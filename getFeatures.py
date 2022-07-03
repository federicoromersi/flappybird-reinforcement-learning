import numpy as np
from build_tiles import build_tiles

def getFeatures(s, a, gridx, gridy, M, N, A):
    
    # numero di celle
    nCells = pow(M, 2)    # sistemare numero celle
    d = A*N*nCells    # dimensione spazio di stato lineare

    # spazio di stato lineare
    state = np.zeros((d, 1))

    for ind in range(0, N):
        xx = gridx[ind, :]  # prendo la componente x della griglia ind-esima
        yy = gridy[ind, :]  # prendo la componente y della griglia ind-esima

        # trovo gli indici dell'elemento di stato s
        ix = find(s[0], xx)
        iy = find(s[1], yy)

        #print(ix, iy)

        # converto lo stato in lineare
        ind = np.ravel_multi_index((ix, iy, ind, a), dims=(M, M, N, A), order='C')
        state[ind] = 1  # assegno 1 agli elementi corrispondenti allo stato

    return state


def find(num, vect):
    for ind in range(0, len(vect)-1):
        if (num >= vect[ind] and num <= vect[ind+1]):
            return ind



