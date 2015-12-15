import random
import numpy as np
NUM_TILES = 4

def ind2sub(ind):
    r = (ind % NUM_TILES)
    c = (ind / NUM_TILES)
    return [int(r), int(c)]

def sub2ind(sub):
    r, c = sub
    return r + NUM_TILES*c

class Agent(object):
    """
    Abstract base class
    """
    def act(self):
        raise NotImplementedError

class RandomAgent(Agent):
    def __init__(self):
        pass

    def act(self, state):
        legal = state.getLegalActions(state.pos)
        action = legal[random.randint(0, len(legal)-1)]
        return action

class VIagent(Agent):
    """
    Agent that performs Value Iteration.
    """
    def __init__(self, mdp):
        self.V = np.zeros(NUM_TILES*NUM_TILES)
        self.mdp = mdp
        pass

    def alg(self, TOL= 0.01):
        c= 0
        oldV = self.V.copy()
        while True:
            for state in range(0, len(self.V)):
                expValue = max(np.dot(self.mdp.Psa[:,state,:], oldV))

                self.V[state] = self.mdp.R[state] + \
                    self.mdp.discount * expValue

            diff = np.linalg.norm(self.V - oldV)
            print(diff)
            if diff < TOL:
                break

            oldV = self.V.copy()

        pass

    def act(self, state):
        pass

class MDP(object):
    def __init__(self, startPos, goalTiles, failTiles, slipTiles, discount = 0.95):
        self.goalTiles = goalTiles
        self.failTiles = failTiles
        self.slipTiles = slipTiles

        self.discount = discount
        self.reward = 0 #Deprecated?
        self.state = State(self, startPos)

        self.Psa = self.createT()
        self.R = self.createR()

    def createR(self):
        """
        Creates the true reward vector for each state
        """
        R = np.zeros(NUM_TILES**2)
        for state in range(0, len(R)):
            if state == 15:
                pass

            if ind2sub(state) in self.failTiles:
                R[state] = -10
            elif ind2sub(state) in self.goalTiles:
                R[state] = 10
            else:
                R[state] = -1

        return R


    def createT(self):
        """
        Creates the true transition probability matrix for the MDP.
        """
        Psa = np.zeros((4, NUM_TILES**2, NUM_TILES**2))

        for state in range(0, NUM_TILES**2):
            if ind2sub(state) in self.failTiles or \
               ind2sub(state) in self.goalTiles:
                continue

            legal = self.state.getLegalActions(ind2sub(state))

            for i, action in enumerate(['up', 'down', 'left', 'right']):
                if action in legal:
                    if ind2sub(state) in self.slipTiles:
                        for slip in legal:
                            succ = sub2ind(self.state.generateSuccessor(ind2sub(state), slip, determine= True))
                            Psa[i, state, succ] = 1.0/len(legal)
                    else:
                        succ = sub2ind(self.state.generateSuccessor(ind2sub(state), action))
                        if succ == 1:
                            pass
                        if state == 1:
                            pass
                        Psa[i, state, succ] = 1

        return Psa

    def updateState(self, action):
        newPos = self.state.generateSuccessor(self.state.pos, action)
        self.state = State(self, newPos)
        self.reward += self.state.getReward()

class State(MDP):
    def __init__(self, mdp, position):
        self.mdp = mdp
        self.pos = position

    def isEnd(self):
        return self.pos in self.mdp.goalTiles or self.pos in self.mdp.failTiles

    def getReward(self):
        reward = 0
        if self.pos in self.mdp.failTiles:
            reward = -10
        elif self.pos in self.mdp.goalTiles:
            reward = 10
        else:
            reward = -1

        return reward

    def generateSuccessor(self, pos, action, determine= False):
        ##ISSUE: Use of determine kw inelegant.
        #Slip occurs
        if pos in self.mdp.slipTiles and not determine:
            alts = self.getLegalActions(pos)
            action = alts[random.randint(0, len(alts)-1)]

        successor = pos.copy()
        if action == 'up':
            successor[0] -= 1
        elif action == 'down':
            successor[0] += 1
        elif action == 'left':
            successor[1] -= 1
        elif action == 'right':
            successor[1] += 1

        assert successor != pos
        return successor

    def getLegalActions(self, pos):
        ##ISSUE: Doesn't make sense to keep this part of "State".
        actions = []
        if pos[0] != NUM_TILES-1:
            actions.append('down')
        if pos[1] != NUM_TILES-1:
            actions.append('right')
        if pos[0] != 0:
            actions.append('up')
        if pos[1] != 0:
            actions.append('left')

        assert actions != []
        return actions

ra = RandomAgent()
mdp = MDP([0, 0], [[NUM_TILES-1, NUM_TILES-1]], [[0, 1],[1, 3]], [[1, 0],[2, 1]])

vi = VIagent(mdp)
vi.alg()

currState = mdp.state
while not currState.isEnd():
    action= ra.act(currState)
    print(action)
    mdp.updateState(action)
    print(mdp.reward)
    currState = mdp.state
    print(currState.pos)





