# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"
        # print("move: ", legalMoves[chosenIndex])
        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        # print("curPos: ", currentGameState.getPacmanPosition())
        # print("newPos: ", newPos)
        # print("newFood: ", newFood.asList())
        # print("newGhostStates: ", newGhostStates)
        # print("newScaredTimes: ", newScaredTimes)
        # print("curScore: ", successorGameState.getScore())

        "*** YOUR CODE HERE ***"
        isClose = 0
        isScared = 0
        ghostStateNum = len(newGhostStates)
        for i in range(ghostStateNum):
            # print("ghost: ", ghostState.getPosition())
            ghostState = newGhostStates[i]
            scaredTime = newScaredTimes[i]
            if abs((ghostState.getPosition()[0] - newPos[0]) + (ghostState.getPosition()[1] - newPos[1])) == 1:
                if scaredTime == 0:
                    isClose = 1
                    break
                else:
                    isScared = 1
                
        if isClose == 1:
            return successorGameState.getScore() - 10
        elif isScared == 1:
            return successorGameState.getScore() + 10
        else:
            return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        numAgents = gameState.getNumAgents()
        actions = []
        inf = 10000000
        maxDepth = self.depth * numAgents

        def value(gameState, depth):
            if gameState.isLose() or gameState.isWin() or depth == maxDepth:
                return self.evaluationFunction(gameState)
            
            if (depth % numAgents == 0):    # if next agent is MAX
                return maxValue(gameState, depth)
            else:                           # if next agent is MIN
                return minValue(gameState, depth)
        
        def maxValue(gameState, depth):
            v = -inf
            agent = 0
            legalMoves = gameState.getLegalActions(agent)
            bestMove = None
            for move in legalMoves:
                nextState = gameState.generateSuccessor(agent, move)
                nextStateValue = value(nextState, depth+1)
                if (nextStateValue > v):
                    v = nextStateValue
                    bestMove = move
            if depth == 0:
                actions.append(bestMove)
            return v

        def minValue(gameState, depth):
            v = inf
            agent = depth % numAgents
            legalMoves = gameState.getLegalActions(agent)
            for move in legalMoves:
                nextState = gameState.generateSuccessor(agent, move)
                nextStateValue = value(nextState, depth+1)
                if (nextStateValue < v):
                    v = nextStateValue

            return v

        value(gameState, 0)
        return actions[-1]


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        numAgents = gameState.getNumAgents()
        actions = []
        inf = 10000000
        maxDepth = self.depth * numAgents
        alpha = -inf
        beta = inf

        def value(gameState, depth, alpha, beta):
            if gameState.isLose() or gameState.isWin() or depth == maxDepth:
                return self.evaluationFunction(gameState)
            
            if (depth % numAgents == 0):    # if next agent is MAX
                return maxValue(gameState, depth, alpha, beta)
            else:                           # if next agent is MIN
                return minValue(gameState, depth, alpha, beta)

        def maxValue(gameState, depth, alpha, beta):
            v = -inf
            agent = 0
            legalMoves = gameState.getLegalActions(agent)
            bestMove = None
            for move in legalMoves:
                nextState = gameState.generateSuccessor(agent, move)
                nextStateValue = value(nextState, depth+1, alpha, beta)
                if (nextStateValue > v):
                    v = nextStateValue
                    bestMove = move
                if (v > beta):
                    return v
                alpha = max(alpha, v)
            if depth == 0:
                actions.append(bestMove)
            return v          

        def minValue(gameState, depth, alpha, beta):
            v = inf
            agent = depth % numAgents
            legalMoves = gameState.getLegalActions(agent)
            for move in legalMoves:
                nextState = gameState.generateSuccessor(agent, move)
                nextStateValue = value(nextState, depth+1, alpha, beta)
                if (nextStateValue < v):
                    v = nextStateValue
                if (v < alpha):
                    return v
                beta = min(beta, v)
            return v   

        value(gameState, 0, alpha, beta)
        return actions[-1]                       


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        numAgents = gameState.getNumAgents()
        actions = []
        inf = 1000000
        maxDepth = numAgents * self.depth

        def value(gameState, depth):
            if gameState.isLose() or gameState.isWin() or depth == maxDepth:
                return self.evaluationFunction(gameState)
            
            if (depth % numAgents == 0):    # if next agent is MAX
                return maxValue(gameState, depth)
            else:                           # if next agent is MIN
                return expValue(gameState, depth)
            
        def maxValue(gameState, depth):
            v = -inf
            agent = 0
            legalMoves = gameState.getLegalActions(agent)
            bestMove = None
            for move in legalMoves:
                nextState = gameState.generateSuccessor(agent, move)
                nextStateValue = value(nextState, depth+1)
                if (nextStateValue > v):
                    v = nextStateValue
                    bestMove = move
            if depth == 0:
                actions.append(bestMove)
            return v

        def expValue(gameState, depth):
            v = 0
            agent = depth % numAgents
            legalMoves = gameState.getLegalActions(agent)
            p = 1 / len(legalMoves)
            for move in legalMoves:
                nextState = gameState.generateSuccessor(agent, move)
                nextStateValue = value(nextState, depth+1)
                v += nextStateValue * p
            return v

        value(gameState, 0)
        return actions[-1]


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    pos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()
    ghostStateNum = len(ghostStates)
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]
    isClose = 0
    isScared = 0
    inf = 10000000
    foodList = food.asList()
    if len(foodList) == 0:
        return inf

    score = currentGameState.getScore()
    for i in range(ghostStateNum):
        ghostState = ghostStates[i]
        scaredTime = scaredTimes[i]
        if (ghostState.getPosition() == pos):
            if scaredTime == 0:
                return -inf
            else:
                score += 100
        if abs((ghostState.getPosition()[0] - pos[0]) + (ghostState.getPosition()[1] - pos[1])) == 1:
            if scaredTime == 0:
                score -= 10
            else:
                score += 10

    dist = 0
    for food in foodList:
        dist += abs((food[0] - pos[0]) + (food[1] - pos[1]))

    return score - dist


# Abbreviation
better = betterEvaluationFunction
