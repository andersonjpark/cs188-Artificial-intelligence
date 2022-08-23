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

        "*** YOUR CODE HERE ***"
        foodList = newFood.asList()
        if action == 'Stop':
            return float("-Inf")
        for ghostState in newGhostStates:
            if (ghostState.getPosition()) is newPos and (ghostState.scaredTimer is False):
                return float("-Inf")

        ghostDistance = set((manhattanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates))
        foodDistance = set((manhattanDistance(newPos, food) for food in foodList))
        if len(foodDistance) == 0:
            return float('Inf')

        return successorGameState.getScore() + min(ghostDistance)/min(foodDistance) + newScaredTimes[0]

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

        Set = set()
        depth = self.depth
        newDepth = depth * gameState.getNumAgents()
        pacmanAction = gameState.getLegalActions(0)

        for action in pacmanAction:
            pacmanSuc = gameState.generateSuccessor(0, action)
            nodeValue = self.minimax(pacmanSuc, newDepth-1, 1)
            Set.add((nodeValue, action))
        return max(Set)[1]

    def minimax(self, gameState, newDepth, agentIndex):
        next = (agentIndex + 1) % gameState.getNumAgents()
        if newDepth < 1 or gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState)

        def minValue(gameState, Depth, agentIndex):
            v = float("+Inf")
            ghostAction = gameState.getLegalActions(agentIndex)
            for gaction in ghostAction:
                ghostSuc = gameState.generateSuccessor(agentIndex, gaction)
                v = min(v, self.minimax(ghostSuc, Depth, next))
            return v

        def maxValue(gameState, Depth, agentIndex):
            v = float("-Inf")
            pacmanAction = gameState.getLegalActions(agentIndex)
            for paction in pacmanAction:
                pacmanSuc = gameState.generateSuccessor(agentIndex, paction)
                v = max(v, self.minimax(pacmanSuc, Depth, next))
            return v

        if agentIndex == 0:
            return maxValue(gameState, newDepth-1, agentIndex)
        if agentIndex != 0:
            return minValue(gameState, newDepth-1, agentIndex)


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        alpha = MAX's best option on path to root
        beta = MIN's best option on path to root
        """
        "*** YOUR CODE HERE ***"
        depth = self.depth
        newDepth = depth * gameState.getNumAgents()

        alpha = float("-inf")
        beta = float("inf")
        return self.alphaBetaPruning(gameState, newDepth, 0, alpha, beta)[1]

    def alphaBetaPruning(self, gameState, newDepth, agentIndex, alpha, beta):
        next = (agentIndex + 1) % gameState.getNumAgents()
        if newDepth == 0 or gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState), None

        def maxPruning(gameState, newDepth, agentIndex, alpha, beta):
            v = float("-Inf")
            v_action = None
            pacmanAction = gameState.getLegalActions(agentIndex)
            for paction in pacmanAction:
                pacmanSuc = gameState.generateSuccessor(agentIndex, paction)
                v_temp = self.alphaBetaPruning(pacmanSuc, newDepth-1, next, alpha, beta)[0]
                if v < v_temp:
                    v_action = paction
                v = max(v, v_temp)
                if v > beta: return v, v_action
                alpha = max(alpha, v)
            return v, v_action

        def minPruning(gameState, newDepth, agentIndex, alpha, beta):
            v = float("+Inf")
            v_action = None
            ghostAction = gameState.getLegalActions(agentIndex)
            for gaction in ghostAction:
                ghostSuc = gameState.generateSuccessor(agentIndex, gaction)
                v_temp = self.alphaBetaPruning(ghostSuc, newDepth-1, next, alpha, beta)[0]
                if v > v_temp:
                    v = v_temp
                    v_action = gaction
                v = min(v, v_temp)
                if v < alpha: return v, v_action
                beta = min(beta, v)
            return v, v_action

        if agentIndex == 0:
            return maxPruning(gameState, newDepth, agentIndex, alpha, beta)
        if agentIndex != 0:
            return minPruning(gameState, newDepth, agentIndex, alpha, beta)

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

        def maxValue(state, newDepth):
            v = float('-Inf')
            legalActions = state.getLegalActions()
            if state.isLose() or state.isWin() or newDepth < 1:
                return self.evaluationFunction(state)
            for action in legalActions:
                temp_v = expectValue(state.generateSuccessor(0, action), newDepth, 1)
                v = max(v, temp_v)
            return v

        def expectValue(state, newDepth, agentIndex):
            v = 0
            if state.isLose() or state.isWin():
                return self.evaluationFunction(state)
            for action in state.getLegalActions(agentIndex):
                if (agentIndex + 1) % state.getNumAgents() != 0:
                    len_ac = len(state.getLegalActions(agentIndex))
                    expV = expectValue(state.generateSuccessor(agentIndex, action), newDepth, agentIndex + 1) / len_ac
                    v += expV
                else:
                    len_act = len(state.getLegalActions(agentIndex))
                    maxV = maxValue(state.generateSuccessor(agentIndex, action), newDepth - 1) / len_act
                    v += maxV
            return v

        def expectimax(state):
            expectiMax = float('-Inf')
            for action in state.getLegalActions(0):
                v = expectValue(state.generateSuccessor(0, action), self.depth, 1)
                if v > expectiMax:
                    rightAction = action
                expectiMax = max(expectiMax, v)
            return rightAction

        return expectimax(gameState)

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: I collected the ghost location and pacmpan location
                    and gave lots of bonus when the pacman scared the ghost
                    and when pacman is far away enough from the ghost
                    (I put disFromGhost least value to 2 since 1 is too risky)
    """
    "*** YOUR CODE HERE ***"
    bonus = 0
    newPos = currentGameState.getPacmanPosition()
    newGhostStates = set(currentGameState.getGhostStates())

    for ghostState in newGhostStates:
        disFromGhost = manhattanDistance(newPos, ghostState.getPosition())
        if ghostState.scaredTimer > 0:
            bonus += 100
        if 2 < disFromGhost < 5:
            bonus += 5
        elif disFromGhost > ghostState.scaredTimer:
            bonus += 50/disFromGhost

    return currentGameState.getScore() + bonus


# Abbreviation
better = betterEvaluationFunction
