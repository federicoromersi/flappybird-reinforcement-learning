import numpy as np
from build_tiles import build_tiles
from play import play
import numpy as np
from numpy import save
from numpy import load
import os.path



def main_linear_tiles():

    lbx = 0.0
    ubx = 2
    lby = -0.6
    uby = 1

    M = 10  # cella 10x10
    N = 4   # numero di griglie per fare l'offset

    A = 2   # numero di azioni possibili

    # numero di celle
    nCells = pow(M, 2)    # sistemare numero celle
    d = A*N*nCells    # dimensione spazio di stato lineare


    [gridx, gridy] = build_tiles(lbx, ubx, lby, uby, M, N)


    if os.path.exists('w.npy'):
        print ("loading w...")
        w = load('w.npy')
    else:
        print ("creating w...")
        w = np.zeros((d, 1))


    nEpisodes = 200000

    eps = 0.1
    for ind in range(0, nEpisodes):
        if ind%3000 == 0:
            w = play(w, gridx, gridy, M, N, A, 1, 0)
        else:
            w = play(w, gridx, gridy, M, N, A, 0, eps)
        
        save('w.npy', w)

        
 




if __name__ == "__main__":
    print("starting...")
    main_linear_tiles()