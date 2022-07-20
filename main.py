from flappy_bird_agent import *
import joblib
from joblib import dump, load

lbx = -0.5 
ubx = 2 
lby = -1 
uby = 1


M = 20  # cella 10x10
N = 4  # numero di griglie per fare l'offset
A = 2

nEpisodes = 500
alpha = 0.001
eps = 0.01


def loadObj():
    if os.path.exists('agent.pkl'):
        agent = joblib.load("agent.pkl")
        if agent.M != M or agent.N != N or agent.nEpisodes != nEpisodes:
            agent = FlappyBirdAgent(nEpisodes, alpha, eps, lbx, ubx, lby, uby, M, N, A)
        else:
            agent.alpha = alpha
            agent.eps = eps
    else:
        agent = FlappyBirdAgent(nEpisodes, alpha, eps, lbx, ubx, lby, uby, M, N, A)

    return agent



if __name__ == "__main__":

    agent = loadObj()

    for i in range(0, 1000):
        print("iter: ", nEpisodes * agent.nIters)

        #agent.plotScores()
        #agent.plot()
        #agent.TC.draw_tiles()

        agent.evaluate(True)
        agent.train()
        
        # salva l'oggetto agent aggiornato nel file
        joblib.dump(agent, "agent.pkl")

        # salve lo score attuale nella lista
        agent.saveScores()

        # plot degli score fino ad adesso
        #agent.plotScores()

        agent.nIters = agent.nIters + 1

