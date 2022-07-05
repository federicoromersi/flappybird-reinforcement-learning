import numpy as np
from numpy import save
from numpy import load
from functions import find
import os.path
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D



class tileCoding():
    def __init__(self, lbx, ubx, lby, uby, M, N, A):
        self.lbx = lbx
        self.ubx = ubx
        self.lby = lby
        self.uby = uby
        self.M = M
        self.N = N
        self.A = A
        self.gridx = np.zeros((M, M))
        self.gridy = np.zeros((M, M))
        self.d = self.A * self.N * self.M * self.M  # dimensione spazio di stato lineare


    def buildTiling(self):
        # offset
        offset = np.array([1, 3])
        #offset = offset/np.amax(offset)

        dx = (self.ubx - self.lbx)/self.M # divido lo spazio di stato in M celle lungo x
        TX =  np.arange(self.lbx - dx, self.ubx, dx)

        dy = (self.uby - self.lby)/self.M # divido lo spazio di stato in M celle lungo y
        TY =  np.arange(self.lby - dy, self.uby, dy)

        print(len(TX))
        print(len(TY))

        # creo le griglie
        gridx = np.zeros((self.N, len(TX)))
        gridy = np.zeros((self.N, len(TY)))

        # creazione griglie con offset
        for ind in range(0, self.N):
            gridx[ind, :] = TX + offset[0]*dx*ind / self.N
            gridy[ind, :] = TY + offset[1]*dy*ind / self.N

        self.gridx = gridx
        self.gridy = gridy


    def getFeatures(self, s, a):

        # spazio di stato lineare
        state = np.zeros((self.d, 1))

        for ind in range(0, self.N):
            xx = self.gridx[ind, :]  # prendo la componente x della griglia ind-esima
            yy = self.gridy[ind, :]  # prendo la componente y della griglia ind-esima

            # trovo gli indici dell'elemento di stato s
            ix = find(s[0], xx)
            iy = find(s[1], yy)

            # converto lo stato in lineare
            ind = np.ravel_multi_index((ix, iy, ind, a), dims=(self.M, self.M, self.N, self.A), order='C')
            state[ind] = 1  # assegno 1 agli elementi corrispondenti allo stato

        return state

    
    def getW(self):

        if os.path.exists('w.npy'):
            print ("loading w...")
            w = load('w.npy')
        else:
            print ("creating w...")
            w = np.zeros((self.d, 1))

        return w

    
    def draw_tiles(self):
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
        linestyles = ['-', '--', ':']
        legend_lines = []

        fig, ax = plt.subplots(figsize=(10, 10))
        for ind in range(0,self.N):
            for x in self.gridx[ind]:
                l = ax.axvline(x=x, color=colors[ind % len(colors)], linestyle=linestyles[ind % len(linestyles)], label=ind)
            for y in self.gridy[ind]:
                l = ax.axhline(y=y, color=colors[ind % len(colors)], linestyle=linestyles[ind % len(linestyles)], label=ind)
            legend_lines.append(l)
        #ax.grid('off')
        ax.legend(legend_lines, ["Tiling #{}".format(t) for t in range(len(legend_lines))], facecolor='white', framealpha=0.9)
        ax.set_title("Tilings")
        plt.show(block=False)

