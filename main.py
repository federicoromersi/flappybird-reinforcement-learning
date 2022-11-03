from curses.ascii import isdigit
from flappy_bird_agent import *
import joblib
from joblib import dump, load
import sys

lbx = -0.5 
ubx = 2 
lby = -1 
uby = 1
lv = -1
uv = 1


M = 8  # cella 10x10
N = 10  # numero di griglie per fare l'offset
A = 2

nEpisodes = 500
alpha = 0.0001
eps = 0.05


def loadObj():
    if os.path.exists('agent.pkl'):
        agent = joblib.load("agent.pkl")
        if agent.M != M or agent.N != N or agent.nEpisodes != nEpisodes:
            agent = FlappyBirdAgent(nEpisodes, alpha, eps, lbx, ubx, lby, uby, lv, uv, M, N, A)
        else:
            agent.alpha = alpha
            agent.eps = eps
    else:
        agent = FlappyBirdAgent(nEpisodes, alpha, eps, lbx, ubx, lby, uby, lv, uv, M, N, A)

    return agent



if __name__ == "__main__":

    if len(sys.argv) >= 2:
        agent_trained = joblib.load("trained_agent.pkl")
        if (sys.argv[1] == "test"):       
            if len(sys.argv) >= 3 and sys.argv[2].isdigit():
                for i in range(0, int(sys.argv[2])):
                    agent_trained.evaluate(False)
            else:
                agent_trained.evaluate(True)

        elif (sys.argv[1] == "scores"):
            agent_trained.plotScores()

        elif (sys.argv[1] == "bestscore"):
            print(agent_trained.bestScore)

        elif (sys.argv[1] == "tiles"):
            agent_trained.TC.draw_tiles()




    else:
        agent = loadObj()
        agent_trained = joblib.load("trained_agent.pkl")

        agent.TC.draw_tiles()
        agent.plotScores()
        agent.evaluate(True)


        for i in range(0, 1000):
            print("iter: ", nEpisodes * agent.nIters)

            agent.train()
            
            # salva l'oggetto agent aggiornato nel file
            joblib.dump(agent, "agent.pkl")

            # salve lo score attuale nella lista
            agent.saveScores()

            if max(agent_trained.scores) <= agent.scores[len(agent.scores)-1]:
                joblib.dump(agent, "trained_agent.pkl")

            agent.nIters = agent.nIters + 1

