import numpy as np
from numpy import save
from numpy import load
from functions import find
import os.path
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as patches




class tileCoding():
    def __init__(self, lbx, ubx, lby, uby, lv, uv, M, N, A):
        self.lbx = lbx
        self.ubx = ubx
        self.lby = lby
        self.uby = uby
        self.lv = lv
        self.uv = uv
        self.M = M
        self.N = N
        self.A = A
        self.gridx = 0
        self.gridy = 0
        self.gridv = 0
        self.d = self.A * self.N * self.M * self.M * self.M # dimensione spazio di stato lineare


    def buildTiling(self):
        # offset
        offset = np.array([1, 3, 1])

        dx = (self.ubx - self.lbx)/(self.M-1) 
        TX = np.linspace(self.lbx - dx, self.ubx, self.M)

        dy = (self.uby - self.lby)/(self.M-1) 
        TY = np.linspace(self.lby - 3*dy, self.uby, self.M)

        dv = (self.uv - self.lv)/(self.M-1) 
        TV = np.linspace(self.lv - dv, self.uv, self.M)

        # creo le griglie
        gridx = np.zeros((self.N, len(TX)))
        gridy = np.zeros((self.N, len(TY)))
        gridv = np.zeros((self.N, len(TV)))

        # creazione griglie con offset
        for ind in range(0, self.N):
            gridx[ind, :] = TX + offset[0]*dx*ind / self.N
            gridy[ind, :] = TY + offset[1]*dy*ind / self.N
            gridv[ind, :] = TV + offset[2]*dv*ind / self.N

        self.gridx = gridx
        self.gridy = gridy
        self.gridv = gridv


    def getFeatures(self, s, a):
        # spazio di stato lineare
        state = np.zeros((self.d, 1))

        for ind in range(0, self.N):
            xx = self.gridx[ind, :]  # prendo la componente x della griglia ind-esima
            yy = self.gridy[ind, :]  # prendo la componente y della griglia ind-esima
            vv = self.gridv[ind, :]

            # trovo gli indici dell'elemento di stato s
            ix = find(s[0], xx)
            iy = find(s[1], yy)
            iv = find(s[2], vv)

            # converto lo stato in lineare
            ind = np.ravel_multi_index((ix, iy, iv, ind, a), dims=(self.M, self.M, self.M, self.N, self.A), order='C')
            state[ind] = 1  # assegno 1 agli elementi corrispondenti allo stato

        return state


    
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
        height = self.uby-self.lby
        width = self.ubx-self.lbx
        rect = patches.Rectangle((self.lbx, self.lby), width, height, linewidth=1, edgecolor='r', facecolor='grey')
        ax.add_patch(rect)
        ax.legend(legend_lines, ["Tiling #{}".format(t) for t in range(len(legend_lines))], facecolor='white', framealpha=0.9)
        ax.set_title("Tilings")
        plt.show(block=False)

