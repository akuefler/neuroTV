import random
import numpy as np
NUM_TILES = 4
NUM_ACTIONS = 4

def ind2sub(ind):
    r = (ind % NUM_TILES)
    c = (ind / NUM_TILES)
    return [int(r), int(c)]

def sub2ind(sub):
    r, c = sub
    return r + NUM_TILES*c

def int2act(integ):
    if integ == 0:
        action = 'up'
    elif integ == 1:
        action = 'down'
    elif integ == 2:
        action = 'left'
    elif integ == 3:
        action = 'right'

    return action

class Agent(object):
    """
    Abstract base class
    """
    def act(self):
        raise NotImplementedError

class RandomAgent(Agent):
    def __init__(self, mdp):
        self.mdp = mdp
        pass

    def act(self, state):
        legal = self.mdp.getLegalActions(state.pos)
        action = legal[random.randint(0, len(legal)-1)]
        return action

class PIagent(Agent):
    def __init__(self, mdp):
        self.V = np.zeros(NUM_TILES*NUM_TILES)
        self.mdp = mdp

        self.pi = np.random.randint(0, NUM_ACTIONS, NUM_TILES**2)

        #Policy Iteration assumes these are known
        self.Psa = self.mdp.Psa
        self.R = self.mdp.R

    def policyEvaluation(self, k= 50):
        V= self.V.copy()

        for i in range(0,k):
            V= [self.R[s] + self.mdp.discount * np.dot(self.Psa[self.pi[s],s,:], V) \
             for s in range(0, NUM_TILES**2)]

        return V


    def policyIteration(self, TOL= 0.01):
        #c= 0
        oldPI = self.pi.copy()
        while True:
            #Set values:
            self.V = self.policyEvaluation()

            #Update policy
            for s in range(0, len(self.V)):
                if s == 12:
                    pass

                legal = self.mdp.getLegalActions(ind2sub(s), returnInts= True)
                u = np.dot(self.Psa[:, s, :], self.V)
                sortu = np.argsort(u)[::-1]
                for idx in sortu:
                    if idx in legal:
                        a = idx
                        break

                self.pi[s] = a

            diff = np.linalg.norm(self.pi - oldPI)
            print(diff)
            if diff < TOL:
                #break
                break

            oldPI = self.pi.copy()

    def act(self, state):
        integ = self.pi[sub2ind(state.pos)]
        return int2act(integ)



class VIagent(Agent):
    """
    Agent that performs Value Iteration.
    """
    def __init__(self, mdp):
        self.V = np.zeros(NUM_TILES*NUM_TILES)
        self.mdp = mdp

        #Value Iteration assumes these are known
        self.Psa = self.mdp.Psa
        self.R = self.mdp.R

    def valueIteration(self, TOL= 0.01):
        oldV = self.V.copy()
        while True:
            for state in range(0, len(self.V)):
                expValue = max(np.dot(self.Psa[:,state,:], oldV))

                self.V[state] = self.R[state] + \
                    self.mdp.discount * expValue

            diff = np.linalg.norm(self.V - oldV)
            print(diff)
            if diff < TOL:
                break

            oldV = self.V.copy()

        pass

    def act(self, state):
        """
        Pick actions as argmax of expected value of values found in value iteration.
        """
        legal = self.mdp.getLegalActions(state.pos, returnInts= True)
        s = sub2ind(state.pos)
        v = np.dot(self.Psa[:, s, :], self.V)
        sortv = np.argsort(v)[::-1]
        for idx in sortv:
            if idx in legal:
                a = idx
                break

        np.argmax(a)

        if a == 0:
            action = 'up'
        elif a == 1:
            action = 'down'
        elif a == 2:
            action = 'left'
        elif a == 3:
            action = 'right'

        return action

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
        Psa = np.zeros((NUM_ACTIONS, NUM_TILES**2, NUM_TILES**2))

        for state in range(0, NUM_TILES**2):
            if ind2sub(state) in self.failTiles or \
               ind2sub(state) in self.goalTiles:
                continue

            legal = self.getLegalActions(ind2sub(state))

            for i, action in enumerate(['up', 'down', 'left', 'right']):
                if action in legal:
                    if ind2sub(state) in self.slipTiles:
                        for slip in legal:
                            succ = sub2ind(self.generateSuccessor(ind2sub(state), slip, determine= True))
                            Psa[i, state, succ] = 1.0/len(legal)
                    else:
                        succ = sub2ind(self.generateSuccessor(ind2sub(state), action))
                        if succ == 1:
                            pass
                        if state == 1:
                            pass
                        Psa[i, state, succ] = 1

        return Psa

    def generateSuccessor(self, pos, action, determine= False):
        ##ISSUE: Use of determine kw inelegant.
        #Slip occurs
        if pos in self.slipTiles and not determine:
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
        assert successor[0] >= 0 and successor[1] >= 0
        return successor

    def getLegalActions(self, pos, returnInts = False):
        ##ISSUE: Doesn't make sense to keep this part of "State".
        actions = []
        actints = []

        if pos[0] != NUM_TILES-1:
            actions.append('down')
            actints.append(1)
        if pos[1] != NUM_TILES-1:
            actions.append('right')
            actints.append(3)
        if pos[0] != 0:
            actions.append('up')
            actints.append(0)
        if pos[1] != 0:
            actions.append('left')
            actints.append(2)

        out = actions
        if returnInts:
            out = actints
        return out

    def updateState(self, action):
        newPos = self.generateSuccessor(self.state.pos, action)
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


#mdp = MDP([0, 0], goalTiles=[[NUM_TILES-1, NUM_TILES-1]], failTiles=[[0, 1],[1, 3]], slipTiles=[[1, 0],[2, 1]])
mdp = MDP([0, 0], goalTiles=[[NUM_TILES-1, 0]], failTiles=[[1, 1],[2, 2],[2, 1],[1, 2]], slipTiles=[[1, 0],[2, 0]])
ra = RandomAgent(mdp)

vi = VIagent(mdp)
vi.valueIteration()

pi = PIagent(mdp)
pi.policyIteration()

currState = mdp.state
while not currState.isEnd():
    #action= ra.act(currState)
    action= pi.act(currState)
    print(action)
    mdp.updateState(action)
    print(mdp.reward)
    currState = mdp.state
    print(currState.pos)





