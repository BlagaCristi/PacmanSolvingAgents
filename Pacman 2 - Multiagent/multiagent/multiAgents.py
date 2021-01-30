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


import random

import util
from game import Agent, Directions, Actions


class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """

    def manhattanDistance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    """
    Usually, an evaluation function should only be interested in the gameState, but, for almost the entire project,
    the evaluation function is also interested in the action taken. The evaluation function is interested in the following
    parameters:
        - if the action is a stopAction, we will apply a penalty for this action. We do not want our agent to stop moving
        unless it really has to stop.
        - the ghost position and if they are scared or not. If we move on a spot in which there is a ghost, we do the following:
            ~ if the ghost is scared, you get a bonus. 
            ~ if the ghost is not scared, you are heavily penalised (this move assures that the game is lost).
        - if we have eaten a food dot, we will apply a bonus
        - we will apply a bonus based on inverse proportionality to the manhattan distances between the pacman position
        and the food dots. Basically, the closer pacman is to food dots, the greater the bonus is. 
        - the last two parameter rules are also applied to the capsules
    All of the weights to these parameters have been chosen by testing different values, whilst also trying to keep a constant
    magnitude between parameters (there are exceptions however, like going over a ghost that is not scared).
    """

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

        foodEaten = 100
        capsuleEaten = 200
        scaredGhostEaten = 300
        intersectWithGhost = -10000

        ifStopped = -1000

        score = 0
        if action == 'Stop':
            score += ifStopped

        # ghostDistances = []
        for ghostState in newGhostStates:
            if newPos == ghostState.getPosition():
                if newScaredTimes[newGhostStates.index(ghostState)] > 0:
                    score += scaredGhostEaten
                else:
                    score += intersectWithGhost
        if currentGameState.hasFood(newPos[0], newPos[1]):
            score += foodEaten

        for food in newFood.asList():
            score += 1.0 / self.manhattanDistance(food, newPos) * 10

        for capsule in currentGameState.getCapsules():
            if newPos == capsule:
                score += capsuleEaten

        for capsule in successorGameState.getCapsules():
            score += 1.0 / self.manhattanDistance(capsule, newPos) * 20

        return score


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
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    """
    The MinimaxAgent is an agent that tries to solve the problem by looking from the perspective
    of the ghost agents, too. The algorithm assumes that all the ghosts play otimally, and, thus, 
    the minmax agent chooses an optimal action too( based on the information that it can rely on).
    That is, the agent can not see, for example, 30 moves ahead. Considering the branching factor ( of 5 = number of actions),
    and the number of levels, we get the following number of gameStates that the algorithm explores (of course, 
    we assume that none of these states is a terminal node):
        5 ** ( pacman (maximizer) + numberOfGhosts (minimizer))*depth
    In other words, for a pacman game with one pacman and 2 ghosts, if we would want to look 30 steps ahead, we would need
    to visit 5 ** 90 gamestates at each step, which is clearly impossible. 
    Thus, the minimaxAgent depth is limited to a small value, in order to assure efficiency of the algorithm. 
    
    The algorithm, however, has some downsides. In some scenarios, it may pacman to kill itself rather than search for a solution
    because it considers it is optimal. In other cases, it may just stop near a food dot because it does not know what to do
    afterwards.
    
    The key idea of the algorithm is that it uses recursion to explore gamestates from different perspectives (maximizer and minimizer),
    and, at the deepest level, evaluate the gamestate with an evaluation function.
    
    A key observation is that a depth of 1 means that both the pacman and the ghosts have performed one move. In terms of the search tree the
    algorithm expands, the tree will look as follows:
    
                        currentGameState
        pacmanMove1     pacmanMove2     pacmanMove3     pacmanMove4
    ghost1Move1 ghost1Move2 .... (5 for each of the previous nodes)
ghost2Move1 ghost2Move2 ... 

    And so on. In terms of depth numbers, if there are k entities (pacman + ghosts), and the current game state has depth -1, we have the following:
    - at depth%k == 0, we have the pacman moves (the sole maximizer level)
    - at depth%k == 1, we have the 1st ghost moves ( the 1st minimizer level)
    - at depth%k == 2, we have the 2nd ghost moves( the 2nd minimizer level)
    and so on.
    
    Thus, if the algorithm searches to a depth h, the search tree will actually have k*h levels.
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
        """
        return self.maxTurn(-1, gameState)[1]

    """
    MaxTurn represents the moment that the pacman moves. On this level, we always choose
    the action that maximizes the profit. 
    """

    def maxTurn(self, actualDepth, gameState):

        actualDepth += 1
        # if we have reached a terminal node, return the utility value
        # return utility value also if we have surpassed the intended depth
        if gameState.isLose() or gameState.isWin() or actualDepth >= self.depth:
            return (self.evaluationFunction(gameState), None)
        else:
            # get Pacman's legal actions
            legalActions = gameState.getLegalActions(0)
            utilityValues = []
            for action in legalActions:
                # for each possible action, we develop the gameState, and call the minTurn function for the 1st ghost
                val, action = self.minTurn(1, actualDepth, gameState.generateSuccessor(0, action))
                utilityValues.append(val)
            return (max(utilityValues), legalActions[utilityValues.index(max(utilityValues))])

    """
    MinTurn represents the moment that the ghosts move. On this level, we always choose
    the action that minimizes the profit. 
    """

    def minTurn(self, ghostAgentIndex, actualDepth, gameState):

        # if we have reached a terminal node, stop
        if gameState.isLose() or gameState.isWin():
            return (self.evaluationFunction(gameState), None)
        # if we have expanded the gamestate for every ghost, go on to the next "depth" level
        # next depth level is represented by pacman performing a new move
        if ghostAgentIndex > gameState.getNumAgents() - 1:
            return self.maxTurn(actualDepth, gameState)
        else:
            legalActions = gameState.getLegalActions(ghostAgentIndex)
            utilityValues = []
            for action in legalActions:
                # for each possible action, we develop the gameState, and call the minTurn function for the next ghost
                val, action = self.minTurn(ghostAgentIndex + 1, actualDepth,
                                           gameState.generateSuccessor(ghostAgentIndex, action))
                utilityValues.append(val)
            return (min(utilityValues), legalActions[utilityValues.index(min(utilityValues))])


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    """
    The AlphaBeta prunning algorithm is basically the Minimax algorithm with one key observation. There may be nodes that do 
    not have to be expanded because the respective states will never be reached ( in the search tree of the algorithm).
    Assume the following search tree:
            Max         A
            Min      B     C
            Max    3  D
                    5   X
            
    Node B, a minimizer node, has as option a value of 3. Node D has explored the leftmost option, and has found a value of 5. At this point, we are
    sure that we do not have to explore the node X. This is because the value chosen by the node D will be at least 5, and we know that node B (a minimizer) 
    already has an option of 3, so it is redundant to check node X. In the same manner, we can work with an upper bound for the minimizer.
    
    We can implement this behavior by using the parameters that each level that are updated with respect to the values returned by their children.
    The solution found by the alphaBetaPrunning agent is exactly the same as the one of the minimax algorithm, the sole difference being that we
    prune actions that we are sure are redundant. 
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        # alpha = value of the best choice found so far for the maximizer
        # beta = value of the best choice found so far for the minimizer
        # this values are consistent along a path from the root node to any node in the search tree
        # "best choice" is relative with respect to the agent's perspective from which we are looking
        self.alpha = -10000
        self.beta = 10000
        return self.maxTurn(-1, gameState, -10000, 10000)[1]

    def maxTurn(self, actualDepth, gameState, alpha, beta):

        actualDepth += 1
        # if we have reached a terminal node, return the utility value
        # return utility value also if we have surpassed the intended depth
        if gameState.isLose() or gameState.isWin() or actualDepth >= self.depth:
            return (self.evaluationFunction(gameState), None)
        else:
            # get Pacman's legal actions
            legalActions = gameState.getLegalActions(0)
            # best option so far
            v = -10000
            bestAction = None
            for action in legalActions:
                val, action1 = self.minTurn(1, actualDepth, gameState.generateSuccessor(0, action), alpha, beta)
                # if we have found an action better than one we have found for this level, we update it
                if v < val:
                    v = val
                    bestAction = action
                # if the action found is greater than the best option for the minimizer, we stop
                if v > beta:
                    return (v, action)
                # otherwise, we update the alpha value ( in case it can be updated)
                alpha = max(v, alpha)
            return (v, bestAction)

    def minTurn(self, ghostAgentIndex, actualDepth, gameState, alpha, beta):

        if gameState.isLose() or gameState.isWin():
            return (self.evaluationFunction(gameState), None)
        if ghostAgentIndex > gameState.getNumAgents() - 1:
            return self.maxTurn(actualDepth, gameState, alpha, beta)
        else:
            legalActions = gameState.getLegalActions(ghostAgentIndex)
            # the best options so far
            v = 10000
            bestAction = None
            for action in legalActions:
                val, action1 = self.minTurn(ghostAgentIndex + 1, actualDepth,
                                            gameState.generateSuccessor(ghostAgentIndex, action), alpha, beta)
                # if we have found an action better than one we have found for this level, we update it
                if val < v:
                    v = val
                    bestAction = action
                # if the action found is greater than the best option for the minimizer, we stop
                if v < alpha:
                    return (v, action)
                # otherwise, we update the beta value ( in case it can be updated)
                if v < beta:
                    beta = v
            return (v, bestAction)


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    """
    The Minimax agent behaves quite well if it assumes that the ghost agents also play optimally. This may represent
    a problem sometimes when the actions the other agent chooses are not necessarily the optimal ones, but rather
    are chosen with a certain probability. 
    In this scenarios, we may upgrade the minimax agent as follows:
        - between each level of agents, we insert a level of "choice" nodes. Basically, the idea is the following:
            ~ we might roll a dice to play backgammon. The value returned has a probability of X. 
                With the value returned by the dice, we might have K options. This representation is expanded from a choice node.
        - the computation of the choice value is not done with min or max values, but rather with an expected value:
            sum (for i from 1 to K) of probability(i) * value (i) 
    The rest of the algorithm is completely the same.
    
    For our game, the probability of an action that the ghost agent can choose is 1/numberOfLegalActions. Each level of choice nodes
    is made of one node actually, thus we can combine the choice node with minLevel. Thus, instead of choosing the minimum of a single value,
    the minLevel will actually compute the expected value.
    
    I have chosen to implement it using only one choice node per choice level because each action has exactly one immediate outcome. 
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        return self.maxTurn(-1, gameState)[1]

    def maxTurn(self, actualDepth, gameState):

        actualDepth += 1
        # if we have reached a terminal node, return the utility value
        # return utility value also if we have surpassed the intended depth
        if gameState.isLose() or gameState.isWin() or actualDepth >= self.depth:
            return (self.evaluationFunction(gameState), None)
        else:
            # get Pacman's legal actions
            legalActions = gameState.getLegalActions(0)
            utilityValues = []
            for action in legalActions:
                val, action = self.minTurn(1, actualDepth, gameState.generateSuccessor(0, action))
                utilityValues.append(val)
            return (max(utilityValues), legalActions[utilityValues.index(max(utilityValues))])

    def minTurn(self, ghostAgentIndex, actualDepth, gameState):

        if gameState.isLose() or gameState.isWin():
            return (self.evaluationFunction(gameState), None)
        if ghostAgentIndex > gameState.getNumAgents() - 1:
            return self.maxTurn(actualDepth, gameState)
        else:
            legalActions = gameState.getLegalActions(ghostAgentIndex)
            utilityValues = []
            for action in legalActions:
                val, action = self.minTurn(ghostAgentIndex + 1, actualDepth,
                                           gameState.generateSuccessor(ghostAgentIndex, action))
                utilityValues.append(val)
            # here is the expected value computed
            return (sum(utilityValues) * 1.0 / len(legalActions), None)


"""
An evaluation function is used in order to assign a value (the higher, the better) to a gameState
with no other information (such as the percept sequence for example).

The evaluation function written for the reflex agent is good for that agent, but not really good 
for the minimax (and its acolytes) algorithm. One one of the reasons, for example, is that it takes into
account the action performed. Thus, I have chosen an approach that starts from that evaluation function, 
but improves some aspects. 

    The parameters we take into account are the following:
    - if the current game state is a win, return a really high value
    - if the current game state is a lose, return a really low value
    - the score is computed by subtracting values. Even if this is counterintuitive, the reason for this
    is that this is a really good way to represent some values, such as distance to closest food dot (
    the smaller the value, the less we subtract). 
    - if we have eaten a food dot, we can see this as follows:
        ~ subtract the number of food dots from the score. If, for example, from 4 actions, only one of them eats
        a food dot, that means that the score of that action is greater with at least 1 from all the scores of the
        other actions ( taking into account only the food dots, with no other constraints). This quantity is weighted 
        with a value ( in our case it's a 10) to express how important eating a food dot is.
    - we also take into account the distance to the closest food dot. This distance is computed by using the BFS 
    algorithm. This value is subtracted from the score. Thus, the smaller the distance, the higher the score is.
    - we also look at the ghost states, but a little bit different than previously. We are going to compensate him
    if there are ghosts that are scared ( implicitly, with this constraint we encourage eating capsules). Also, 
    we are going to give pacman a penalty based on the distance from him to the ghosts ( the closer he is, the worse).
    This was a tricky thing to do, because we do not want to have to much computation time spent in this evaluation 
    function ( so I have chosen to use manhattan distance rather than actual distance to the ghosts), and we also have 
    to find a good way to quantitatively represent this information.
        The last part was the hardest thing to do. I have first started with the way I represented the information in 
    the previous evaluation function, but this thing made pacman to have a low score. By varying the respective values, 
    it still did not prove to be more efficient. So i have decided to chose another approach. We still want to give him a
    penalty based on the distance ( the closer, the worse), but we will do it by subtracting a small part of the distance,
    rather than working with the inverse of the distance. With a little bit of tweaking all the quantitative values, I
    have come up with an evaluation function that the autograder evaluates to the maximum grade ( based on 10 games
    played, pacman has a score slightly over 1000, but never less). 
"""


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    if currentGameState.isWin():
        return 10000
    elif currentGameState.isLose():
        return -10000

    pacmanPosition = currentGameState.getPacmanPosition()
    foodPositionMatrix = currentGameState.getFood()
    wallPositionMatrix = currentGameState.getWalls()
    ghostStates = currentGameState.getGhostStates()

    score = -10 * len(foodPositionMatrix.asList())

    score -= breadthFirstSearch(pacmanPosition, wallPositionMatrix, foodPositionMatrix.asList())

    for ghostState in ghostStates:
        if ghostState.scaredTimer > 0:
            score += 10
        else:
            score -= 0.2 * manhatanDistance(pacmanPosition, ghostState.getPosition())

    return score


def breadthFirstSearch(pacmanPosition, wallPosition, goalPosition):
    from util import Queue
    queue = Queue()
    queue.push(pacmanPosition)
    visited = { pacmanPosition: (None, None) }
    solution = []
    while not queue.isEmpty() and not solution:
        node = queue.pop()
        if node in goalPosition:
            solution = computeSolution(node, visited)
        if not solution:
            for direction in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
                x, y = node
                dx, dy = Actions.directionToVector(direction)
                nextx, nexty = int(x + dx), int(y + dy)
                if not wallPosition[nextx][nexty]:
                    if (nextx, nexty) not in visited:
                        visited[(nextx, nexty)] = (node, direction)
                        queue.push((nextx, nexty))
    return len(solution)


def computeSolution(node, visited):
    solution = []
    while True:
        if visited[node][1]:
            solution.append(visited[node][1])
            node = visited[node][0]
        else:
            break
    # solution has to be reversed because it is computed starting from the goal
    solution.reverse()
    return solution


def manhatanDistance(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


# Abbreviation
better = betterEvaluationFunction
