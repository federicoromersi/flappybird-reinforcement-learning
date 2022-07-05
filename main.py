from flappy_bird_agent import *


lbx = -0.5
ubx = 2
lby = -1
uby = 1


#lbx = 0.0
#ubx = 2
#lby = -0.6
#uby = 1

M = 18  # cella 10x10
N = 4   # numero di griglie per fare l'offset
A = 2

nEpisodes = 5000
alpha = 0.001
eps = 0.01


if __name__ == "__main__":
    agent = FlappyBirdAgent(nEpisodes, alpha, eps, lbx, ubx, lby, uby, M, N, A)
    for i in range(0, 1000):
        agent.evaluate()
        agent.train()
