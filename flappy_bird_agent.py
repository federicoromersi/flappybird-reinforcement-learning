from this import s
import time
import gym 
import flappy_bird_gym
import numpy as np
import random as rd
from tile_coding import *


class FlappyBirdAgent():
    def __init__(self, nEpisodes, alpha, eps, lbx, ubx, lby, uby, M, N, A):
        self.nEpisodes = nEpisodes
        self.lbx = lbx
        self.ubx = ubx
        self.lby = lby
        self.uby = uby
        self.M = M
        self.N = N
        self.A = A
        self.lamb = 0.98
        self.env = flappy_bird_gym.make("FlappyBird-v0")
        self.alpha = alpha
        self.eps = eps
        self.TC = tileCoding(lbx, ubx, lby, uby, M, N, A)
        self.w = self.TC.getW()
        
        self.TC.buildTiling()
        self.TC.draw_tiles()


    def epsGreedy(self, s):
        if rd.uniform(0, 1) < self.eps:
            a = rd.randint(0, self.A-1)
        else:
            q = np.zeros((self.A, 1))
            for a in range(0, self.A):
                w_tran = np.transpose(self.w)
                state = self.TC.getFeatures(s, a)
                q[a] = np.dot(w_tran, state)
            a = np.argmax(q)
        return a


    def run(self):
        obs = self.env.reset()
        action = 0
        score = 0
        s, reward, done, info = self.env.step(action)
        while (not done):

            state = self.TC.getFeatures(s, action)

            sp, reward, done, info = self.env.step(action)

            if score < info["score"]:
                score = info["score"]
                pipeReward = 1
            else:
                pipeReward = 0


            if done:
                self.w = self.w + self.alpha*(-10 - np.dot(np.transpose(self.w), state)) * state
                print(score)
            else:
                ap = self.epsGreedy(sp)   # scegli azione successiva
                statep = self.TC.getFeatures(sp, ap)  # vai allo stato successivo
                self.w = self.w + self.alpha*(score + np.dot(np.transpose(self.w), statep) - np.dot(np.transpose(self.w), state)) * state
                
            s = sp
            action = ap

            if self.eps == 0:
                self.env.render()
                time.sleep(1 / 200)

    
    def runET(self):
        obs = self.env.reset()
        z = np.zeros((self.TC.d, 1))
        action = 0
        score = 0
        s, reward, done, info = self.env.step(action)
        while (not done):

            state = self.TC.getFeatures(s, action)

            sp, reward, done, info = self.env.step(action)

            if score < info["score"]:
                score = info["score"]
                pipeReward = 1
            else:
                pipeReward = 0


            if done:
                delta = - np.dot(np.transpose(self.w), state)
                print(score)
            else:
                ap = self.epsGreedy(sp)   # scegli azione successiva
                statep = self.TC.getFeatures(sp, ap)  # vai allo stato successivo
                delta = reward + np.dot(np.transpose(self.w), statep) - np.dot(np.transpose(self.w), state)

    
            z = self.lamb * z + state
            self.w = self.w + self.alpha * delta * z
            s = sp
            action = ap

            if self.eps == 0:
                self.env.render()
                time.sleep(1 / 200)
            
            



    
    def evaluate(self):
        obs = self.env.reset()
        action = 0
        score = 0
        s, reward, done, info = self.env.step(action)
        while (not done):

            state = self.TC.getFeatures(s, action)

            sp, reward, done, info = self.env.step(action)

            q = np.zeros((self.A, 1))
            for a in range(0, self.A):
                w_tran = np.transpose(self.w)
                state = self.TC.getFeatures(s, a)
                q[a] = np.dot(w_tran, state)
            ap = np.argmax(q)

            s = sp
            action = ap

            self.env.render()
            time.sleep(1 / 200)

    
    def train(self):
        if self.eps == 0:
            self.runET()
        else:
            for ind in range(0, self.nEpisodes):
                self.runET()
                save('w.npy', self.w)


        
     



    