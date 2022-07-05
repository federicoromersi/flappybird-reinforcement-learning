import time
import gym 
import numpy as np
import flappy_bird_gym

from epsGreedy import epsGreedy
from getRBF import getRBF



def playRBF(w, gridx, gridy, M, N, A, check, eps):

    alpha = 0.01
    sigma = 0.01

    env = flappy_bird_gym.make("FlappyBird-v0")

    obs = env.reset()

    action = 0
    s, reward, done, info = env.step(action)
    while (not done):

        state = getRBF(s, action, sigma, gridx, gridy, M, N, A)

        sp, reward, done, info = env.step(action)

        score = info["score"]


        if done:
            w = w + alpha*(-10 - np.dot(np.transpose(w), state)) * state
        else:
            ap = epsGreedy(sp, w, eps, gridx, gridy, M, N, A)   # scegli azione successiva
            statep = getRBF(s, action, sigma, gridx, gridy, M, N, A)  # vai allo stato successivo
            w = w + alpha*(score*1 + np.dot(np.transpose(w), statep) - np.dot(np.transpose(w), state)) * state
            

        s = sp
        action = ap

        if check == 1:
            env.render()
            time.sleep(1 / 200)

    
    return w

