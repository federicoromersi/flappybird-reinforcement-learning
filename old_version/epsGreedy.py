import random as rd
import numpy as np
from getFeatures import getFeatures

def epsGreedy(s, w, eps, gridx, gridy, M, N, A):

    if rd.uniform(0, 1) < eps:
        a = rd.randint(0, A-1)
    else:
        q = np.zeros((A, 1))
        for a in range(0, A):
            w_tran = np.transpose(w)
            state = getFeatures(s, a, gridx, gridy, M, N, A)
            q[a] = np.dot(w_tran, state)
        a = np.argmax(q)

    return a