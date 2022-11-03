import time
import gym 
import flappy_bird_gym
import numpy as np
import random as rd
from tile_coding import *


class FlappyBirdAgent():
    def __init__(self, nEpisodes, alpha, eps, lbx, ubx, lby, uby, lv, uv, M, N, A):
        self.nEpisodes = nEpisodes
        self.lbx = lbx
        self.ubx = ubx
        self.lby = lby
        self.uby = uby
        self.lv = lv
        self.uv = uv
        self.M = M
        self.N = N
        self.A = A
        self.lamb = 0.98 #0.98
        self.env = flappy_bird_gym.make("FlappyBird-v0")
        self.alpha = alpha
        self.eps = eps
        self.TC = tileCoding(lbx, ubx, lby, uby, lv, uv, M, N, A)
        self.w = np.zeros((self.A * self.N * self.M * self.M * self.M, 1))
        self.nIters = 0
        self.scores = []
        self.bestScore = 0

        self.TC.buildTiling()



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


    def greedy(self, s):
        q = np.zeros((self.A, 1))
        for a in range(0, self.A):
            w_tran = np.transpose(self.w)
            state = self.TC.getFeatures(s, a)
            q[a] = np.dot(w_tran, state)
        a = np.argmax(q)
        return a

    
    def runET(self):
        obs = self.env.reset()
        z = np.zeros((self.TC.d, 1))
        action = 0
        score = 0
        s, reward, done, info = self.env.step(action)

        s_est = np.concatenate([s, [0]])

        while (not done):

            state = self.TC.getFeatures(s_est, action)

            sp, reward, done, info = self.env.step(action)

            velp = sp[1] - s[1]

            sp_est = np.concatenate([sp, [velp]])

            if score < info["score"]:
                score = info["score"]
                pipeReward = 1
            else:
                pipeReward = 0

            if done:
                R =  -10     
            else:
                R =  pipeReward*10*score + 0.01    


            if done:
                delta = R - np.dot(np.transpose(self.w), state)
                print(score)
            else:
                ap = self.epsGreedy(sp_est)   # scegli azione successiva
                statep = self.TC.getFeatures(sp_est, ap)  # vai allo stato successivo
                delta = R + np.dot(np.transpose(self.w), statep) - np.dot(np.transpose(self.w), state)

            z = self.lamb * z + state
            self.w = self.w + self.alpha * delta * z
            s_est = sp_est
            action = ap

        self.env.close()


              
    def evaluate(self, render):
        obs = self.env.reset()
        z = np.zeros((self.TC.d, 1))
        action = 0
        score = 0
        s, reward, done, info = self.env.step(action)

        while (not done):

            sp, reward, done, info = self.env.step(action)

            velp = sp[1] - s[1]

            sp_est = np.concatenate([sp, [velp]])

            if score < info["score"]:
                score = info["score"]
                if score > self.bestScore:
                    self.bestScore = score
            
            if done:
                print(score)
            else:
                ap = self.greedy(sp_est)   # scegli azione successiva

            s_est = sp_est
            action = ap

            if (render == True):
                self.env.render()
                time.sleep(1 / 100)
        
        self.env.close()

        return score
        

    
    def train(self):
        for ind in range(0, self.nEpisodes):
            self.runET()


    def saveScores(self):
        iters = 50
        sumScore = 0
        for ind in range(0, iters):
            sumScore = sumScore + self.evaluate(False)
        
        meanScore = round(sumScore / iters)

        self.scores.append(meanScore)



    
    def plotScores(self):
        x = range(0, len(self.scores)) 
        x = [i * self.nEpisodes for i in x]
        fig = plt.figure()
        plt.plot(x, self.scores)
        plt.show(block=False)



    