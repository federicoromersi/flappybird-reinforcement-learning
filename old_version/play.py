import time
import gym 
import numpy as np
import flappy_bird_gym

from epsGreedy import epsGreedy
from getFeatures import getFeatures



def play(w, gridx, gridy, M, N, A, checkRender, eps):

    alpha = 0.0001
    env = flappy_bird_gym.make("FlappyBird-v0")

    obs = env.reset()

    action = 0
    score = 0
    s, reward, done, info = env.step(action)
    while (not done):

        state = getFeatures(s, action, gridx, gridy, M, N, A)

        sp, reward, done, info = env.step(action)

        if score < info["score"]:
            score = info["score"]
            pipeReward = 1
        else:
            pipeReward = 0



        if done:
            w = w + alpha*(- 10 - np.dot(np.transpose(w), state)) * state
        else:
            ap = epsGreedy(sp, w, eps, gridx, gridy, M, N, A)   # scegli azione successiva
            statep = getFeatures(sp, ap, gridx, gridy, M, N, A)  # vai allo stato successivo
            w = w + alpha*(score + np.dot(np.transpose(w), statep) - np.dot(np.transpose(w), state)) * state
            

        s = sp
        action = ap

        if checkRender == 1:
            env.render()
            time.sleep(1 / 200)

    
    return w

