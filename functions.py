def find(num, vect):
    for ind in range(0, len(vect)-1):
        if (num >= vect[ind] and num <= vect[ind+1]):
            return ind


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