NUM_TILES = 5
import random

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
        legal = state.getLegalActions()
        action = legal[random.randint(0, len(legal)-1)]
        return action


class MDP(object):
    def __init__(self, startPos= None, goalTiles= None, failTiles= None, slipTiles= None):
        self.goalTiles = goalTiles
        self.failTiles = failTiles
        self.slipTiles = slipTiles

        self.reward = 0
        self.state = State(self, startPos)

    def updateState(self, action):
        newPos = self.state.generateSuccessor(action)
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
            reward = -50
        elif self.pos in self.mdp.goalTiles:
            reward = 50
        else:
            reward = -1

        return reward

    def generateSuccessor(self, action):
        #Slip occurs
        if self.pos in self.mdp.slipTiles and random.random() > 0.5:
            alts = self.getLegalActions()
            alts.remove(action)
            action = alts[random.randint(0, len(alts)-1)]

        successor = self.pos.copy()
        if action == 'up':
            successor[0] -= 1
        elif action == 'down':
            successor[0] += 1
        elif action == 'left':
            successor[1] -= 1
        elif action == 'right':
            successor[1] += 1

        assert successor != self.pos
        return successor

    def getLegalActions(self):
        actions = []
        if self.pos[0] != NUM_TILES-1:
            actions.append('down')
        if self.pos[1] != NUM_TILES-1:
            actions.append('right')
        if self.pos[0] != 0:
            actions.append('up')
        if self.pos[1] != 0:
            actions.append('left')

        assert actions != []
        return actions

ra = RandomAgent()
mdp = MDP([0, 0], [NUM_TILES-1, NUM_TILES-1], [[0, 1],[1, 3]], [[1, 0],[2, 1]])

currState = mdp.state
while not currState.isEnd():
    action= ra.act(currState)
    print(action)
    mdp.updateState(action)
    print(mdp.reward)
    currState = mdp.state
    print(currState.pos)





