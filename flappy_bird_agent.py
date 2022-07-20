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
        self.w = np.zeros((self.A * self.N * self.M * self.M, 1))
        self.nIters = 0
        self.scores = []
        self.avgReward = 0

        self.iA = 1000000 * np.eye(self.TC.d)
        self.B = np.zeros((self.TC.d, 1))
        
        self.TC.buildTiling()
        #self.plot()
        #self.TC.draw_tiles()
        #self.plotScores()


    def epsGreedy(self, s):
        if rd.uniform(0, 1) < self.eps:
            a = rd.randint(0, self.A-1)
            #if rd.uniform(0, 1) < 0.01:
            #    a = 1
            #else:
            #    a = 0
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
        
        self.env.close()


    
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
                R =  -10      # - self.avgReward
            else:
                R =  pipeReward*10 + 0.01     # - self.avgReward


            if done:
                delta = R - np.dot(np.transpose(self.w), state)
                print(score)
            else:
                ap = self.epsGreedy(sp)   # scegli azione successiva
                statep = self.TC.getFeatures(sp, ap)  # vai allo stato successivo
                delta = R + np.dot(np.transpose(self.w), statep) - np.dot(np.transpose(self.w), state)

            self.avgReward = self.avgReward + 0.001*delta

            z = self.lamb * z + state
            self.w = self.w + self.alpha * delta * z
            s = sp
            action = ap

        self.env.close()


    def runQET(self):
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
                delta = -10 - np.dot(np.transpose(self.w), state)
                print(score)
            else:
                q = np.zeros((self.A, 1))
                for a in range(0, self.A):
                    w_tran = np.transpose(self.w)
                    statep = self.TC.getFeatures(sp, a)
                    q[a] = np.dot(w_tran, statep)
                ap = np.argmax(q)   # scegli azione successiva

                statep = self.TC.getFeatures(sp, ap)  # vai allo stato successivo
                delta = 5*pipeReward + reward*0.1 + np.dot(np.transpose(self.w), statep) - np.dot(np.transpose(self.w), state)

    
            z = self.lamb * z + statep
            self.w = self.w + self.alpha * delta * z
            s = sp
            action = self.epsGreedy(sp)

        self.env.close()


    
    def runLS(self):
        obs = self.env.reset()
        action = 0
        score = 0
        s, reward, done, info = self.env.step(action)
        iA = self.iA
        B = self.B

        while (not done):

            if score < info["score"]:
                score = info["score"]
                pipeReward = 1
            else:
                pipeReward = 0


            state = self.TC.getFeatures(s, action)
            sp, reward, done, info = self.env.step(action)

            ap = self.epsGreedy(sp) 
            statep = self.TC.getFeatures(sp, ap)

            iA = iA - np.dot(iA, state) * np.dot(np.transpose(state - statep), iA) / (1 + np.dot(np.dot(np.transpose((state - statep)), iA), state))
            
            if done:
                R = -10
                print(score)
            else:
                R = pipeReward*10 + 0.01

            
            B = B + R*state
            self.w = np.dot(iA, B)

            s = sp
            action = ap

        self.iA = iA
        self.B = B

        self.env.close()

    
    def runNStepSarsa(self):
        obs = self.env.reset()
        action = 0
        score = 0
        s, reward, done, info = self.env.step(action)

        z = np.zeros((self.TC.d, 1))
        Qold = 0
        while (not done):

            state = self.TC.getFeatures(s, action)
            sp, reward, done, info = self.env.step(action)

            if score < info["score"]:
                score = info["score"]
                pipeReward = 1
            else:
                pipeReward = 0

            if done:
                R = -10
                print(score)
            else:
                R = pipeReward*5 + 0.01
            
            ap = self.epsGreedy(sp)
            statep = self.TC.getFeatures(sp, ap)

            Q = self.w.T @ state
            Qp = self.w.T @ statep

            delta = R + Qp - Q

            z = self.lamb * z + (1 - self.alpha * self.lamb * (z.T @ state))*state
            self.w = self.w + self.alpha*(delta + Q - Qold)*z - self.alpha*(Q - Qold)*state

            Qold = Qp
            
            s = sp
            action = ap
        
        self.env.close()



                  

    
    def evaluate(self, render):
        obs = self.env.reset()
        action = 0
        score = 0
        s, reward, done, info = self.env.step(action)
        while (not done):

            sp, reward, done, info = self.env.step(action)

            q = np.zeros((self.A, 1))

            if score < info["score"]:
                score = info["score"]
                pipeReward = 1
            else:
                pipeReward = 0

            for a in range(0, self.A):
                w_tran = np.transpose(self.w)
                statep = self.TC.getFeatures(sp, a)
                q[a] = np.dot(w_tran, statep)
            ap = np.argmax(q)

            s = sp
            action = ap

        

            if (render == True):
                self.env.render()
                time.sleep(1 / 300)


        self.env.close()

        return score
        

    
    def train(self):
        for ind in range(0, self.nEpisodes):
            self.runLS()


    
    def plot(self):
        dx = (self.ubx - self.lbx)/self.M 
        dy = (self.uby - self.lby)/self.M 
        x = np.arange(self.lbx, self.ubx, dx)
        y = np.arange(self.lby, self.uby, dy)

        Z = np.zeros([len(x), len(y)])
        for i in range(0, len(x)):
            for j in range(0, len(x)):
                s = [x[i], y[j]]
                q = np.zeros(self.A)
                for a in range(0, self.A):
                    q[a] = np.dot(np.transpose(self.w), self.TC.getFeatures(s, a))

                Z[i,j] = np.max(q)
        
        fig = plt.figure()
        ax = fig.gca(projection="3d")
        X, Y = np.meshgrid(x, y)
        surf = ax.plot_surface(X, Y, Z, cmap=plt.get_cmap())
        plt.show(block=False)


    def saveScores(self):

        iters = 10
        sumScore = 0
        for ind in range(0, iters):
            sumScore = sumScore + self.evaluate(False)
        
        meanScore = round(sumScore / iters)

        self.scores.append(meanScore)


    
    def plotScores(self):
        x = range(0, len(self.scores)) 
        x = [i * self.nEpisodes for i in x]
        plt.plot(x, self.scores)
        plt.show(block=False)



    