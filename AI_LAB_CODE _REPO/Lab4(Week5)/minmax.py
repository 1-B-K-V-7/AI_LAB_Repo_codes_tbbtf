import numpy as np
import time

# Define the Node class to represent a state in the game tree
class Node():

    def __init__(self, state, depth, maximiser, player, parent=None):
        self.state = state
        self.children = list()
        self.score = 0
        self.depth = depth
        self.maximiser = maximiser
        self.player = player
        self.best_child = None

        if parent is not None:
            parent.children.append(self)

    def __hash__(self):
        # Define a hash function based on the state for hashing purposes
        return hash(str(self.state))

# Define the Environment class to represent the game environment
class Environment():

    def __init__(self, start_state=None):
        if start_state is None:
            # Initialize the start state if not provided
            self.start_state = np.array([['.','.','.'],['.','.','.'],['.','.','.']])
        else:
            self.start_state = start_state

    # Method to get possible moves from a given state for a player
    def get_moves(self, state, player):
        new_states = []
        for i in range(3):
            for j in range(3):
                if state[i][j]=='.':
                    new_state = state.copy()
                    new_state[i,j] = player
                    new_states.append(new_state)
        return new_states

    # Method to check if a given state is a terminal state
    def check_terminal(self, state):
        for i in range(3):
            for j in range(3):
                if state[i][j]=='.':
                    return False
        return True

    # Method to evaluate the utility of a given state
    def evaluate(self, state):
        for val in range(3):
            if state[val,0] == state[val,1] == state[val,2]!='.':
                if state[val, 0]=='x':
                    return 1
                else:
                    return -1
            if state[0,val] == state[1,val] == state[2,val]!='.':
                if state[0,val]=='x':
                    return 1
                else:
                    return -1
        if state[0,0] == state[1,1] == state[2,2]!='.':
            if state[0,0]=='x':
                return 1
            else:
                return -1
        if state[0,2] == state[1,1] == state[2,0]!='.':
            if state[0,2]=='x':
                return 1
            else:
                return -1
        return 0

    # Method to get the start state of the environment
    def get_start_state(self):
        return self.start_state

# Define the Agent class to implement the Minimax algorithm
class Agent():

    def __init__(self, env, maximiser):
        self.env = env
        self.start_state = env.get_start_state()
        self.root_node = None
        self.neginf = -10**18
        self.posinf = 10**18
        self.maximiser = maximiser
        self.player = 'x' if maximiser else 'o';
        self.explored_nodes = 0

    # Minimax algorithm implementation
    def minimax(self, node):
        self.explored_nodes+=1
        score = self.env.evaluate(node.state)
        if score!=0:
            node.score = score
            return node
        if self.env.check_terminal(node.state):
            node.score = 0
            return node
        if node.maximiser:
            best_score = self.neginf
            best_depth = self.posinf
            next_moves = self.env.get_moves(node.state, node.player)
            for move in next_moves:
                child = Node(state = move, depth=node.depth+1,
                             maximiser=not node.maximiser, player='o', parent=node)
                child= self.minimax(child)
                node.children.append(child)
                if best_score<child.score or (best_score==child.score and child.depth<best_depth):
                    best_score = child.score
                    best_depth = child.depth
                    node.best_child = child
            node.depth = best_depth
            node.score = best_score
            return node
        else:
            best_score = self.posinf
            best_depth = self.posinf
            next_moves = self.env.get_moves(node.state, node.player)
            for move in next_moves:
                child = Node(state = move, depth=node.depth+1,
                             maximiser=not node.maximiser, player='x', parent=node)
                child = self.minimax(child)
                node.children.append(child)
                if best_score>child.score  or (best_score==child.score and child.depth<best_depth):
                    best_score = child.score
                    best_depth = child.depth
                    node.best_child = child
            node.depth = best_depth
            node.score = best_score
            return node

    # Method to run the Minimax algorithm
    def run(self):
        self.root_node = Node(state=self.start_state, depth=0, maximiser=self.maximiser,
                             player=self.player, parent=None)
        self.root_node = self.minimax(node = self.root_node)

    # Method to print the states traversed by the Minimax algorithm
    def print_nodes(self):
        node = self.root_node
        while node is not None:
            print(node.state)
            node = node.best_child

# Define the AlphaBetaAgent class to implement the Alpha-Beta Pruning algorithm
class AlphaBetaAgent():

    def __init__(self, env, maximiser):
        self.env = env
        self.start_state = env.get_start_state()
        self.root_node = None
        self.neginf = -10**18
        self.posinf = 10**18
        self.maximiser = maximiser
        self.player = 'x' if maximiser else 'o';
        self.explored_nodes = 0

    # Alpha-Beta Pruning algorithm implementation
    def minimax(self, node, alpha, beta):
        self.explored_nodes+=1
        score = self.env.evaluate(node.state)
        if score!=0:
            node.score = score
            return node
        if self.env.check_terminal(node.state):
            node.score = 0
            return node
        if node.maximiser:
            best_score = self.neginf
            best_depth = self.posinf
            next_moves = self.env.get_moves(node.state, node.player)
            for move in next_moves:
                child = Node(state = move, depth=node.depth+1,
                             maximiser=not node.maximiser, player='o', parent=node)
                child= self.minimax(node = child, alpha=alpha, beta=beta)
                node.children.append(child)
                if best_score<child.score or (best_score==child.score and child.depth<best_depth):
                    best_score = child.score
                    best_depth = child.depth
                    node.best_child = child
                alpha = max(alpha, best_score)
                if beta <= alpha:
                    break
            node.depth = best_depth
            node.score = best
            return node
        else:
            best_score = self.posinf
            best_depth = self.posinf
            next_moves = self.env.get_moves(node.state, node.player)
            for move in next_moves:
                child = Node(state = move, depth=node.depth+1,
                             maximiser=not node.maximiser, player='x', parent=node)
                child = self.minimax(node = child, alpha=alpha, beta=beta)
                node.children.append(child)
                if best_score>child.score  or (best_score==child.score and child.depth<best_depth):
                    best_score = child.score
                    best_depth = child.depth
                    node.best_child = child
                beta = min(beta, best_score)
                if beta<=alpha:
                    break
            node.depth = best_depth
            node.score = best_score
            return node

    # Method to run the Alpha-Beta Pruning algorithm
    def run(self):
        self.root_node = Node(state=self.start_state, depth=0, maximiser=self.maximiser,
                             player=self.player, parent=None)
        self.root_node = self.minimax(node = self.root_node, alpha = self.neginf, beta = self.posinf)

    # Method to print the states traversed by the Alpha-Beta Pruning algorithm
    def print_nodes(self):
        node = self.root_node
        while node is not None:
            print(node.state)
            node = node.best_child

# Sample usage of the Minimax Agent
start_state = np.array([['.','.','.'],['.','.','.'],['.','.','.']])
env = Environment(start_state = start_state)
agent = Agent(env, maximiser=True)
agent.run()
agent.print_nodes()

# Sample usage of the Alpha-Beta Pruning Agent
start_state = np.array([['X','.','.'],['.','.','.'],['.','.','.']])
env = Environment(start_state = start_state)
agent = AlphaBetaAgent(env, maximiser=True)
agent.run()
agent.print_nodes()

# Testing the performance of Minimax Agent
t = 0
for i in range(10):
    start_state = np.array([['.','.','.'],['.','.','.'],['.','.','.']])
    env = Environment(start_state = start_state)
    agent = Agent(env, maximiser=True)
    start_time = time.time()
    agent.run()
    end_time = time.time()
    t += end_time-start_time
print("Average time required for Minimax Agent:", t/10)

# Testing the performance of Alpha-Beta Pruning Agent
t = 0
for i in range(10):
    start_state = np.array([['.','.','.'],['.','.','.'],['.','.','.']])
    env = Environment(start_state = start_state)
    agent = AlphaBetaAgent(env, maximiser=True)
    start_time = time.time()
    agent.run()
    end_time = time.time()
    t += end_time-start_time
print("Average time required for Alpha-Beta Pruning Agent:", t/10)

# Getting the number of nodes explored by Alpha-Beta Pruning Agent
agent = AlphaBetaAgent(env, maximiser=True)
agent.run()
print("Number of nodes explored by Alpha-Beta Pruning Agent:", agent.explored_nodes)
