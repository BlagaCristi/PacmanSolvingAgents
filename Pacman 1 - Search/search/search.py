# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


"""
Function used in order to compute the solution of a search algorithm
based on a dictionary of type:
    dictionary[node]=(parentNode, action)
For the root of the tree that is built by the search algorithm, 
both the action and the parentNode is None.
"""


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


"""
Depth first search algorithm -> an uninformed search algorithm.
The base idea of the algorithm is the following one:
    -> if you were to imagine the search tree that a search algorithm
    can expand, the DFS algorithm always chooses the leftmost option
Algorithm is implemented based on a recursive solution.
"""


def depthFirstSearch(problem):
    # Initialization part of the search Algorithm
    from util import Stack
    stack = Stack()
    stack.push(problem.getStartState())

    """ 
    This dictionary will help both in reconstructing the solution and in keeping track of the already visited nodes
    With the help of this dictionary we also avoid loops
    Structure: dict[node] = (parent, move)
    """
    visited = { problem.getStartState(): (None, None) }
    solution = depthFirstSearchRecursive(problem, stack, visited)

    return solution


def depthFirstSearchRecursive(problem, stack, visited):
    # take the top of the stack and put it into the node variable
    node = stack.pop()
    solution = []
    # if the node is a goal state, we compute the solution
    if problem.isGoalState(node):
        solution = computeSolution(node, visited)
    else:
        """
        DFS is a deterministic algorithm but behaves differently based on the order 
        in which the neighbour list of a node is given. In some cases, certain path lengths may differ 
        (as is the case for the MediumMaze) depending on whether the neighbour list is reversed or not.
        """
        # take each of the successors of the current node and add it to the stack if not visited yet
        for successor in problem.getSuccessors(node):
            if successor[0] not in visited and not solution:
                stack.push(successor[0])
                visited[successor[0]] = (node, successor[1])
                # each node is explored in the moment of discovery
                solution = depthFirstSearchRecursive(problem, stack, visited)
    return solution


"""
Breadth first search algorithm is an uniformed search method that uses a frontier(implemented as a queue)
to store the nodes that represent the bound between the visited nodes and the nodes to be visited.
If we consider that each step has a cost of 1 ( that is, each edge has a weight of 1), the algorithm
finds the shortest path from the root node to the goal node.
"""


def breadthFirstSearch(problem):
    # initialization part of the algorithm
    from util import Queue
    queue = Queue()
    queue.push(problem.getStartState())
    visited = { problem.getStartState(): (None, None) }
    solution = []
    while not queue.isEmpty() and not solution:
        # extract a node from the queue
        node = queue.pop()
        # if the node is a goal state, then compute solution
        if problem.isGoalState(node):
            solution = computeSolution(node, visited)
        if not solution:
            # take each successor of the current node and add it to the queue if not visited
            for successor in problem.getSuccessors(node):
                if successor[0] not in visited:
                    visited[successor[0]] = (node, successor[1])
                    queue.push(successor[0])
    return solution


"""
Uniform Cost Search is an uninformed search algorithm that behaves similarly to the Breadth Frist Search algorithm,
with two key differences (it is important to note that the UCS algorithm behaves optimally for any step cost function,
whereas BFS assures optimality only for a step cost function of 1). 
    The first difference is that it uses a PriorityQueue (the most prioritary node is the minimum one)
instead of a Queue. The PriorityQueue elements are the following ones ( they have been changed from the original
implementation in order to allow a more flexible behaviour):
    ((node, pathToNodeSoFar),priority)
    The second difference is that the UCS algorithm always chooses to expand the node that is the closest to the root
node, thus assuring optimality of the solution. Also, when a node that was already visited is encountered, we chose not
to ignore it, but rather to see if it can be updated (that is, the path we are on now is actually less costly than the
one we have encountered it for the 1st time).
    One key observation of the UCS algorithm is that when an element is popped from the PriorityQueue, the optimal solution
to that node has already been found and any route to that node from that moment on will be less optimal.
    Another key observation is that the algorithm works only for non-negative step cost functions. Negative costs would throw
the algorithm in an infinite loop.
"""


def uniformCostSearch(problem):
    # initialization part of the algorithm
    from util import PriorityQueue
    priorityQueue = PriorityQueue()
    priorityQueue.push((problem.getStartState(), []), 0)
    solution = []
    visited = []
    while not priorityQueue.isEmpty() and not solution:
        # take the element with the most priority and put it into elem
        elem = priorityQueue.pop()
        node = elem[0]
        # only when we pop the element we are sure we have reached it OPTIMALLY
        visited.append(node[0])
        priority = elem[1]
        # if the node found is a goal state, we return the solution
        if problem.isGoalState(node[0]):
            solution = node[1]
        else:
            # for each successor of the node, if it has not been visited ( it has not been popped from the PQ), add it to the PQ
            for successor in problem.getSuccessors(node[0]):
                if successor[0] not in visited:
                    movementList = node[1][:]
                    movementList.append(successor[1])
                    # the element is "updated" in the queue:
                    # - if it's not present, add it
                    # - if it's present, update the priority in case the new one is lower
                    priorityQueue.update((successor[0], movementList), priority + successor[2])
    return solution


def nullHeuristic(state, problem = None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


"""
    AStarSearch if an INFORMED search algorithm. It behaves exactly as the UCS algorithm, with one key difference
that is essential. The UCS algorithm orders the elements in the PQ based solely on the cost of the path from 
the root to the actual node. Even if the UCS algorithm assures optimality, it still searches a lot of the search-space
with no knowledge whatsoever about the knowledge of the goal. In order to incorporate this knowledge in an algorithm, 
we can make use of a heuristic function that represents a lower bound on the actual distance from the current state
to the goal.
    Thus, by changing the priority function from the distance so far to:
        priority(node) = distance(node) + heuristic(node)
    we will prune a lot of the search space.
    One key observation is that in order for the A* search algorithm to be optimal, the heuristic has to be
consistent (and therefore admissible).
    Another observation is that each element of the PQ is interpreted as follows:
        ((node, pathFromRootToNode),priority,actualDistance)
"""


def aStarSearch(problem, heuristic = nullHeuristic):
    from util import PriorityQueue
    priorityQueue = PriorityQueue()
    # for the root node, the priority is 0 (actual distance from the root to the root) plus the heuristic (approximate distance to the goal)
    priorityQueue.push((problem.getStartState(), [], 0), heuristic(problem.getStartState(), problem))
    solution = []
    visited = []
    while not priorityQueue.isEmpty() and not solution:
        elem = priorityQueue.pop()
        node = elem[0]
        # only when we pop the element we are sure we have reached it OPTIMALLY
        visited.append(node[0])
        actualDistance = node[2]
        if problem.isGoalState(node[0]):
            solution = node[1]
        else:
            for successor in problem.getSuccessors(node[0]):
                if successor[0] not in visited:
                    movementList = node[1][:]
                    movementList.append(successor[1])
                    priorityQueue.update((successor[0], movementList, actualDistance + successor[2]),
                                         actualDistance + successor[2] + heuristic(successor[0], problem))
    return solution


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
