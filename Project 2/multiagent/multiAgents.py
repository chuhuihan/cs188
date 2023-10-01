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
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
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

    def evaluationFunction(self, currentGameState: GameState, action):
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
        
        "Takes in current position, proposed next position"
        "Want to get away from the ghosts, get closer to the pellets"

        "find closest pellet w min(manhattan distance)"
        "find closest ghost w min(manhattan distance)"

        score = 0

        newGhostPositions = successorGameState.getGhostPositions()
        newCapsulePositions = successorGameState.getCapsules()

        distanceFromGhosts = []
        distanceFromFood = []
        distanceFromCapsules = []

        for position in newGhostPositions:
            distanceFromGhosts.append(manhattanDistance(newPos, position))

        amountOfDotsRemaining = 0

        for rowInd, row in enumerate(newFood):
                for colInd, obj in enumerate(row):
                    if(obj == True): 
                        amountOfDotsRemaining += 1
                        xy = [rowInd, colInd]
                        distanceFromFood.append(manhattanDistance(newPos, xy))

        for position in newCapsulePositions:
            distanceFromCapsules.append(manhattanDistance(newPos, position))

        #Getting min distances
        closestGhost = min(distanceFromGhosts)
        if(len(distanceFromCapsules) > 0):
            closestCapsule = min(distanceFromCapsules)
        else: 
            closestCapsule = 0

        if(len(distanceFromFood) > 0):
            closestFood = min(distanceFromFood)
        else:
            closestFood = 0

        #Score Calculations
        score += 50 / (closestFood + 1)

        if(closestGhost < 3):
            score += -100
        else:
            score += 10 / (closestGhost + 1)
        
        score += -25 * amountOfDotsRemaining

        if((closestCapsule == 0)):
            score += 200
        else:
            score += 50 / (closestCapsule +1)

        return score

        "return successorGameState.getScore()"


def scoreEvaluationFunction(currentGameState: GameState):
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

    def getAction(self, gameState: GameState):
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
        #util.raiseNotDefined()

        # def miniMax(self, new_gameState, agentIndex, depth):
        #     if(depth == 0 or new_gameState.isWin() or new_gameState.isLose()):
        #         return self.evaluationFunction(new_gameState)
        #     elif(agentIndex == 0):
        #         legalActions = new_gameState.getLegalActions(agentIndex)
        #         branches = []
        #         for action in legalActions: 
        #             branches.append(new_gameState.generateSuccessor(agentIndex, action))
        #         all_moves = [ miniMax(self, b, (agentIndex + 1) % new_gameState.getNumAgents(), depth - 1)   for b in branches  ]
        #         if (depth == self.depth):
        #             return legalActions[all_moves.index(max(all_moves))]
        #         else:
        #             return max(all_moves)
        #     else:
        #         legalActions = new_gameState.getLegalActions(agentIndex)
        #         branches = []
        #         for action in legalActions:
        #             branches.append(new_gameState.generateSuccessor(agentIndex, action))
        #         all_moves = [ miniMax(self, b, (agentIndex + 1) % new_gameState.getNumAgents(), depth - 1)   for b in branches  ]
        #         if (depth == self.depth):
        #             return legalActions[all_moves.index(min(all_moves))]
        #         else:
        #             return min(all_moves)

        # return miniMax(self, gameState, 0, self.depth)

        def miniMax(self, new_gameState, agentIndex, depth):

            if(depth == 0 or new_gameState.isWin() or new_gameState.isLose()):
                return self.evaluationFunction(new_gameState)
            
            next_depth = depth
            if(agentIndex == new_gameState.getNumAgents() - 1):
                next_depth -= 1

            if(agentIndex == 0):
                legalActions = new_gameState.getLegalActions(agentIndex)
                branches = []
                for action in legalActions: 
                    branches.append(new_gameState.generateSuccessor(agentIndex, action))
                all_moves = [ miniMax(self, b, (agentIndex + 1) % new_gameState.getNumAgents(), next_depth)   for b in branches  ]
                if (self.depth == depth):
                    return legalActions[all_moves.index(max(all_moves))]
                return max(all_moves)
            else:
                legalActions = new_gameState.getLegalActions(agentIndex)
                branches = []
                for action in legalActions:
                    branches.append(new_gameState.generateSuccessor(agentIndex, action))
                all_moves = [ miniMax(self, b, (agentIndex + 1) % new_gameState.getNumAgents(), next_depth)   for b in branches  ]
                return min(all_moves)

        return miniMax(self, gameState, 0, self.depth)
        

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        
        # def maxValue(state, a, beta, agentIndex):
        #     v = float('-inf')
        #     legalActions = state.getLegalActions(agentIndex)
        #     branches = []
        #     for action in legalActions:
        #         branches.append(state.generateSuccessor(agentIndex, action))
            
        #     for successor in branches:
        #         eval = 
        #         v = max(v, )

        # def minValue(state, a, beta):

        # def alphaBetaMiniMax(self, new_gameState, agentIndex, depth, alpha, beta):
            
        #     # check if depth is at 0 or if game is ending
        #     if(depth == 0 or new_gameState.isWin() or new_gameState.isLose()):
        #         return self.evaluationFunction(new_gameState)

        #     next_depth = depth

        #     # decrements depth if we are on the last ghost
        #     if(agentIndex == new_gameState.getNumAgents() - 1):
        #         next_depth -= 1
            
        #     # maximizing (pacman)
        #     if(agentIndex == 0):
                
        #         best_score = float('-inf')
        #         # get all of states of the legal actions for pacman at this state and put it into branches
        #         legalActions = new_gameState.getLegalActions(agentIndex)
        #         branches = []
        #         for action in legalActions:
        #             branches.append(new_gameState.generateSuccessor(agentIndex, action))
                
        #         # all_moves = []
        #         # recurse into each state and get its score/prune it
        #         for state in branches:
        #             evalScore = alphaBetaMiniMax(self, state, (agentIndex + 1) % new_gameState.getNumAgents(), next_depth, alpha, beta)
        #             best_score = max(best_score, evalScore)
        #             if (best_score >= beta):
        #                 break
        #             alpha = max(alpha, best_score)
        #         # if(self.depth == depth):
        #         #     return legalActions[all_moves.index(max(all_moves))]
        #         return best_score
            
        #     else:

        #         best_score = float('inf')

        #         # get all of states of the legal actions for pacman at this state and put it into branches
        #         legalActions = new_gameState.getLegalActions(agentIndex)
        #         branches = []
        #         for action in legalActions:
        #             branches.append(new_gameState.generateSuccessor(agentIndex, action))
                
        #         # recurse into each state and get its score/prune it
        #         for state in branches:
        #             evalScore = alphaBetaMiniMax(self, state, (agentIndex + 1) % new_gameState.getNumAgents(), next_depth, alpha, beta)
        #             best_score = min(best_score, evalScore)
        #             beta = min(beta, best_score)
        #             if (best_score <= alpha):
        #                 break
        #             beta = min(beta, best_score)
        #         return best_score
        
        # infinity = float('inf')
        # negInfinity = float('-inf')
        
        # return alphaBetaMiniMax(self, gameState, 0, self.depth, negInfinity, infinity)

        # def alphaBetaMiniMax(self, new_gameState, agentIndex, depth, alpha, beta):
            
        #     # check if depth is at 0 or if game is ending
        #     if(depth == 0 or new_gameState.isWin() or new_gameState.isLose()):
        #         return self.evaluationFunction(new_gameState)
            
        #     # maximizing (pacman)
        #     if(agentIndex == 0):
                
        #         best_score = float('-inf')
        #         # get all of states of the legal actions for pacman at this state and put it into branches
        #         legalActions = new_gameState.getLegalActions(agentIndex)
        #         for action in legalActions:
        #             childState = new_gameState.generateSuccessor(agentIndex, action)
        #             best_score = max(best_score, alphaBetaMiniMax(self, childState, (agentIndex + 1) % new_gameState.getNumAgents(), depth, alpha, beta))
        #             alpha = max(alpha, best_score)
        #             if (beta <= alpha):
        #                 break
        #         # if(self.depth == depth):
        #         #     return legalActions[all_moves.index(max(all_moves))]
        #         return best_score
            
        #     else:

        #         best_score = float('inf')

        #         # get all of states of the legal actions for pacman at this state and put it into branches
        #         legalActions = new_gameState.getLegalActions(agentIndex)
        #         for action in legalActions:
        #             childState = new_gameState.generateSuccessor(agentIndex, action)
        #             # if ghost is the last agent
        #             if(agentIndex == new_gameState.getNumAgents()):
        #                 best_score = min(best_score, alphaBetaMiniMax(self, childState, (agentIndex + 1) % new_gameState.getNumAgents(), depth - 1, alpha, beta))
        #             else:
        #                 best_score = min(best_score, alphaBetaMiniMax(self, childState, (agentIndex + 1) % new_gameState.getNumAgents(), depth, alpha, beta))
        #             beta = min(beta, best_score)
        #             if (beta <= best_score):
        #                 break
        #         return best_score
        
        # infinity = float('inf')
        # negInfinity = float('-inf')
        
        # return alphaBetaMiniMax(self, gameState, 0, self.depth, negInfinity, infinity)

        def alphaBetaMiniMax(self, new_gameState, agentIndex, depth, alpha, beta):
            
            # check if depth is at 0 or if game is ending
            if(depth == 0 or new_gameState.isWin() or new_gameState.isLose()):
                return self.evaluationFunction(new_gameState)

            next_depth = depth

            # decrements depth if we are on the last ghost
            if(agentIndex == new_gameState.getNumAgents() - 1):
                next_depth -= 1
            
            # maximizing (pacman)
            if(agentIndex == 0):
                best_score = float('-inf')
                legalActions = new_gameState.getLegalActions(agentIndex)
                all_moves = []
                for action in legalActions:
                    next_state = new_gameState.generateSuccessor(agentIndex, action)
                    curr_score = alphaBetaMiniMax(self, next_state, (agentIndex + 1) % new_gameState.getNumAgents(), next_depth, alpha, beta)
                    best_score = max(best_score, curr_score)
                    alpha = max(alpha, best_score)
                    if alpha > beta:
                        break
                    all_moves.append(curr_score)
                if self.depth == depth:
                    return legalActions[all_moves.index(max(all_moves))]
                return best_score


            else:
                best_score = float('inf')
                legalActions = new_gameState.getLegalActions(agentIndex)
                for action in legalActions:
                    next_state = new_gameState.generateSuccessor(agentIndex, action)
                    curr_score = alphaBetaMiniMax(self, next_state, (agentIndex + 1) % new_gameState.getNumAgents(), next_depth, alpha, beta)
                    best_score = min(best_score, curr_score)
                    beta = min(best_score, curr_score)
                    if alpha > beta:
                        break
                return best_score
        
        return alphaBetaMiniMax(self, gameState, 0, self.depth, float('-inf'), float('inf'))
        


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()

        def miniMax(self, new_gameState, agentIndex, depth):

            if(depth == 0 or new_gameState.isWin() or new_gameState.isLose()):
                return self.evaluationFunction(new_gameState)
            
            next_depth = depth
            if(agentIndex == new_gameState.getNumAgents() - 1):
                next_depth -= 1

            if(agentIndex == 0):
                legalActions = new_gameState.getLegalActions(agentIndex)
                branches = []
                for action in legalActions: 
                    branches.append(new_gameState.generateSuccessor(agentIndex, action))
                all_moves = [ miniMax(self, b, (agentIndex + 1) % new_gameState.getNumAgents(), next_depth)   for b in branches  ]
                if (self.depth == depth):
                    return legalActions[all_moves.index(max(all_moves))]
                return max(all_moves)
            else:
                legalActions = new_gameState.getLegalActions(agentIndex)
                branches = []
                for action in legalActions:
                    branches.append(new_gameState.generateSuccessor(agentIndex, action))
                all_moves = [ miniMax(self, b, (agentIndex + 1) % new_gameState.getNumAgents(), next_depth)   for b in branches  ]
                return float(sum(all_moves)) / len(all_moves)

        return miniMax(self, gameState, 0, self.depth)

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    # util.raiseNotDefined()

    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    score = 0

    newGhostPositions = currentGameState.getGhostPositions()
    newCapsulePositions = currentGameState.getCapsules()

    distanceFromGhosts = []
    distanceFromFood = []
    distanceFromCapsules = []

    for position in newGhostPositions:
        distanceFromGhosts.append(manhattanDistance(newPos, position))

    amountOfDotsRemaining = 0

    for rowInd, row in enumerate(newFood):
            for colInd, obj in enumerate(row):
                if(obj == True): 
                    amountOfDotsRemaining += 1
                    xy = [rowInd, colInd]
                    distanceFromFood.append(manhattanDistance(newPos, xy))

    for position in newCapsulePositions:
        distanceFromCapsules.append(manhattanDistance(newPos, position))

    #Getting min distances
    closestGhost = min(distanceFromGhosts)
    if(len(distanceFromCapsules) > 0):
        closestCapsule = min(distanceFromCapsules)
    else: 
        closestCapsule = 0

    amountCapsulesRemaining = len(newCapsulePositions)
    if(len(distanceFromFood) > 0):
        closestFood = min(distanceFromFood)
    else:
        closestFood = 0

    #Score Calculations
    score += -50 * amountOfDotsRemaining
    score += -100 * amountCapsulesRemaining

    if (max(newScaredTimes) > 0):
        if closestGhost < 3:
            score += max(newScaredTimes)*10 / (closestGhost + 1)
        else:
            score += max(newScaredTimes)*5 / (closestGhost + 1)
    else:
        if(closestGhost < 2):
            score += -10000

    if not closestFood == 0:
        score += 50 / (closestFood + 1)

    if closestCapsule > 0:
        score += 100 / (closestCapsule + 1)

    score += 500 / currentGameState.getScore()

    return score

# Abbreviation
better = betterEvaluationFunction
