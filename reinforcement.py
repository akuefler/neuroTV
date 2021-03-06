import random
import numpy as np
import copy
import math

NUM_TILES = 4
NUM_ACTIONS = 4

FAIL = -10
WIN = 10
NEUTRAL = -1

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

def act2int(act):
    if act == 'up':
        i = 0
    elif act == 'down':
        i = 1
    elif act == 'left':
        i = 2
    elif act == 'right':
        i = 3

    return i

class Agent(object):
    """
    Abstract base class
    """
    def __init__(self, mdp):
        self.V = np.zeros(NUM_TILES*NUM_TILES)
        self.mdp = mdp

        self.policy = np.ones(NUM_TILES**2)*np.nan
        for s in range(0, NUM_TILES**2):
            legal = self.mdp.getLegalActions(ind2sub(s), returnInts= True)
            self.policy[s] = np.random.choice(legal)

    def act(self, state, epsilon = 0):
        if np.random.rand() > epsilon:
            a = self.policy[sub2ind(state.pos)]
        else:
            legal= self.mdp.getLegalActions(state.pos, returnInts = True)
            a = np.random.choice(legal)
        return int2act(a)


    def updatePolicy(self, s, u):
            legal = self.mdp.getLegalActions(ind2sub(s), returnInts= True)
            sortu = np.argsort(u)[::-1]
            for idx in sortu:
                if idx in legal:
                    a = idx
                    break

            self.policy[s] = a


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

        self.policy = np.ones(NUM_TILES**2)*np.nan
        for s in range(0, NUM_TILES**2):
            legal = self.mdp.getLegalActions(ind2sub(s), returnInts= True)
            self.policy[s] = np.random.choice(legal)

        #Policy Iteration assumes these are known
        self.Psa = self.mdp.Psa
        self.R = self.mdp.R

    def policyEvaluation(self, k= 50):
        V= self.V.copy()

        for i in range(0,k):
            V= [self.R[s] + self.mdp.discount * np.dot(self.Psa[self.policy[s],s,:], V) \
             for s in range(0, NUM_TILES**2)]

        return V

    def policyIteration(self, TOL= 0.01):
        #c= 0
        oldPI = self.policy.copy()
        while True:
            #Set values:
            self.V = self.policyEvaluation()

            #Update policy
            for s in range(0, len(self.policy)):
                u = np.dot(self.Psa[:, s, :], self.V)
                self.updatePolicy(s, u)

            diff = np.linalg.norm(self.policy - oldPI)
            if diff < TOL:
                break

            oldPI = self.policy.copy()


class VIagent(Agent):
    """
    Agent that performs Value Iteration.
    """
    def __init__(self, mdp):
        self.V = np.zeros(NUM_TILES*NUM_TILES)
        self.mdp = mdp

        self.policy = np.ones(NUM_TILES**2)*np.nan

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
            if diff < TOL:
                break

            oldV = self.V.copy()

        #Set policy based on values.
        for s in range(0, len(self.V)):
            legal = self.mdp.getLegalActions(ind2sub(s),returnInts= True)
            v = np.dot(self.Psa[:, s, :], self.V)
            sortv = np.argsort(v)[::-1]
            for idx in sortv:
                if idx in legal:
                    a = idx
                    break

            self.policy[s] = a



class MCagent(PIagent):
    """
    Monte Carlo reinforcement learning.

    ISSUE: Should not inherit PIagent because PIagent knows Psa and R a priori.
    """
    def estimate(self, k= 50, epsilon= 0.2):

        C = np.zeros((NUM_ACTIONS, NUM_TILES**2, NUM_TILES**2))
        R = np.zeros(NUM_TILES**2)

        for i in range(0, k):
            ##ISSUE: Can I do this with an mdp method? e.g., "simulate"
            mdp = copy.deepcopy(self.mdp)
            currState = mdp.state
            while not currState.isEnd():
                s = copy.deepcopy(currState)
                action = self.act(s, epsilon = epsilon)
                #print(action)
                mdp.updateState(action)
                currState = mdp.state

                C[act2int(action), sub2ind(s.pos), sub2ind(currState.pos)] += 1
                R[sub2ind(currState.pos)] = currState.getReward() #Assume deterministic reward.

            Psa = np.zeros(C.shape)
            #Normalize counts to obtain transition matrix.
            for a in range(0,Psa.shape[0]):
                for s in range(0,Psa.shape[1]):
                    if sum(C[a,s,:])>0:
                        Psa[a,s,:] = C[a,s,:]/sum(C[a,s,:])

            self.Psa = Psa
            self.R = R
            self.policyIteration()


class MFMCagent(Agent):
    """
    Just computes a Q matrix for a given policy.
    """
    def estimate(self, k = 5000, epsilon= 0):
        ##ISSUE: Once I have Q's, don't know what to use them for...

        Q = np.zeros((NUM_ACTIONS, NUM_TILES**2))
        updates = np.zeros((NUM_TILES**2, NUM_ACTIONS))

        for i in range(0, k):
            ##ISSUE: Can I do this with an mdp method? e.g., "simulate"
            mdp = copy.deepcopy(self.mdp)
            currState = mdp.state

            utility = 0
            t = 0
            while not currState.isEnd():
                state = copy.deepcopy(currState)
                action = self.act(state, epsilon = epsilon)
                #print(action)
                mdp.updateState(action)
                currState = mdp.state

                #Update u
                utility += (mdp.discount**t)*currState.getReward()
                t += 1
                s = sub2ind(state.pos)
                a = act2int(action)

                #Define learning rate
                updates[s,a] += 1
                eta = 1.0/math.sqrt(updates[s, a])
                #eta = 1.0/(1 + updates[s, a]) ##ISSUE: Not sure if I want t here.

                #Update Q
                ##ISSUE: Should I wait to end of episode?
                Q[a,s] -= eta*(Q[a, s] - utility)

        self.Q = Q

class SARSAagent(Agent):
    def estimate(self, k = 5000, epsilon= 0):
        Q = np.zeros((NUM_ACTIONS, NUM_TILES**2))
        updates = np.zeros((NUM_TILES**2, NUM_ACTIONS))

        for i in range(0, k):
            ##ISSUE: Can I do this with an mdp method? e.g., "simulate"
            mdp = copy.deepcopy(self.mdp)
            currState = mdp.state

            utility = 0
            t = 0
            while True:
                state = copy.deepcopy(currState)
                action = self.act(state, epsilon = epsilon)
                mdp.updateState(action)
                currState = mdp.state

                #Update u
                s = sub2ind(state.pos)
                a = act2int(action)
                r = currState.getReward()
                s_ = sub2ind(currState.pos)
                a_ = act2int(self.act(currState, epsilon= epsilon))

                #Define learning rate
                updates[s,a] += 1
                eta = 1.0/math.sqrt(updates[s, a])

                #Update Q
                Q[a, s] += eta*(r + mdp.discount*Q[a_, s_] - Q[a, s])

                if currState.isEnd():
                    break

        self.Q = Q

class QLagent(Agent):
    def estimate(self, k = 5000, epsilon= 0):
        Qopt = np.zeros((NUM_ACTIONS, NUM_TILES**2))
        updates = np.zeros((NUM_TILES**2, NUM_ACTIONS))

        for i in range(0, k):
            mdp = copy.deepcopy(self.mdp)
            currState = mdp.state

            utility = 0
            t = 0
            while True:
                state = copy.deepcopy(currState)
                action = self.act(state, epsilon = epsilon)
                mdp.updateState(action)
                currState = mdp.state

                ##Update u
                s = sub2ind(state.pos)
                a = act2int(action)
                r = currState.getReward()

                #Define learning rate
                updates[s,a] += 1
                eta = 1.0/math.sqrt(updates[s, a])

                #Update Q
                Vopt = max(Qopt[:, sub2ind(currState.pos)])
                Qopt[a, s] += eta*(r + mdp.discount*Vopt - Qopt[a, s])

                if currState.isEnd():
                    break

        #Update policy based on Qopt
        for s in range(0, len(self.policy)):
            q = Qopt[:,s]
            self.updatePolicy(s, q)


class MDP(object):
    def __init__(self, startPos, goalTiles, failTiles, slipTiles, discount = 0.95):
        self.goalTiles = goalTiles
        self.failTiles = failTiles
        self.slipTiles = slipTiles

        self.discount = discount
        self.reward = 0 #Deprecated?
        self.state = State(self, startPos)

        self.Psa = self.createPsa()
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
                R[state] = FAIL
            elif ind2sub(state) in self.goalTiles:
                R[state] = WIN
            else:
                R[state] = NEUTRAL

        return R


    def createPsa(self):
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
        ##ISSUE: Because "self" is never used, maybe this should be its own function entirely.

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
        assert self.state.isEnd() == False

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
        ##ISSUE: Perhaps redundant, given R in mdp. But this is nicer if I don't want an agent to ever touch "R".
        reward = 0
        if self.pos in self.mdp.failTiles:
            reward = FAIL
        elif self.pos in self.mdp.goalTiles:
            reward = WIN
        else:
            reward = NEUTRAL

        return reward


#mdp = MDP([0, 0], goalTiles=[[NUM_TILES-1, NUM_TILES-1]], failTiles=[[0, 1],[1, 3]], slipTiles=[[1, 0],[2, 1]])
mdp = MDP([0, 0], goalTiles=[[NUM_TILES-1, 0]], failTiles=[[1, 1],[2, 2],[2, 1],[1, 2]], slipTiles=[[1, 0],[2, 0]])
ra = RandomAgent(mdp)

vi = VIagent(mdp)
vi.valueIteration()

pi = PIagent(mdp)
pi.policyIteration()

mc = MCagent(mdp)
mc.estimate(k = 50, epsilon= 0.2)
mc.policyIteration()

mfmc = MFMCagent(mdp)
mfmc.policy = pi.policy
mfmc.estimate(k = 5000, epsilon= 0.2)
#mfmc.policyIteration()

sarsa = SARSAagent(mdp)
sarsa.policy = pi.policy
sarsa.estimate(k= 5000, epsilon= 0.2)

ql = QLagent(mdp)
ql.estimate(k= 5000, epsilon= 1)

currState = mdp.state
while not currState.isEnd():
    #action= ra.act(currState)
    action= ql.act(currState)
    print(action)
    mdp.updateState(action)
    print(mdp.reward)
    currState = mdp.state
    print(currState.pos)
