import numpy as np


def build_tiles(lbx, ubx, lby, uby, M, N):

    # offset
    off = np.array([1, 3])
    off = off/np.amax(off)

    dx = (ubx-lbx)/M # divido lo spazio di stato in M celle lungo x
    TX =  np.arange(lbx - dx, ubx, dx)

    dy = (uby-lby)/M # divido lo spazio di stato in M celle lungo y
    TY =  np.arange(lby - dy, uby, dy)

    
    # creo le griglie
    gridx = np.zeros((N, len(TX)))
    gridy = np.zeros((N, len(TY)))

    # creazione griglie con offset
    for ind in range(0, N):
        gridx[ind, :] = TX + off[0]*dx/N*ind
        gridy[ind, :] = TY + off[1]*dy/N*ind


    return [gridx, gridy]
    


