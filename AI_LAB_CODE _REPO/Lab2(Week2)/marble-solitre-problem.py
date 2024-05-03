import copy
import time
import sys
from collections import deque
import heapq

class PriorityQueue():
    def __init__(self):
        self.elements = []
        self.index = 0

    def put(self, elem, priority):
        heapq.heappush(self.elements, (-priority, self.index, elem))
        self.index += 1

    def get(self):
        return heapq.heappop(self.elements)[2]

    def __contains__(self, key):
        for elem in self.elements:
            if elem[1] == key:
                return True
        return False

    def empty(self):
        return not self.elements

# Define game board representations
initial_board = [0, 0, 2, 2, 2, 0, 0,
                 0, 2, 2, 2, 2, 1, 0,
                 1, 2, 1, 1, 1, 2, 2,
                 1, 1, 1, 2, 1, 2, 2,
                 1, 1, 2, 2, 1, 2, 2,
                 0, 2, 1, 2, 1, 1, 0,
                 0, 0, 1, 2, 1, 0, 0]

goal_board = [0, 0, 2, 2, 2, 0, 0,
              0, 2, 2, 1, 2, 2, 0,
              2, 2, 2, 2, 2, 2, 2,
              2, 2, 2, 2, 2, 2, 2,
              2, 2, 2, 2, 2, 2, 2,
              0, 2, 2, 2, 2, 2, 0,
              0, 0, 2, 2, 2, 0, 0]

# Define peg indices for the board legend
legend = [0, 1, 2, 3, 4, 5, 6,
          7, 8, 9, 10, 11, 12, 13,
          14, 15, 16, 17, 18, 19, 20,
          21, 22, 23, 24, 25, 26, 27,
          28, 29, 30, 31, 32, 33, 34,
          35, 36, 37, 38, 39, 40, 41,
          42, 43, 44, 45, 46, 47, 48]

# Define game state class
class GameState():
    def __init__(self, board, goal, x_size, y_size, pegs):
        self.board = board
        self.goal = goal
        self.x_size = x_size
        self.y_size = y_size
        self.children = 0
        self.pegs = pegs
        self.isolated = 0
        self.parent = None
        if self.parent is not None:
            self.pegs = pegs - 1
            self.isolated = parent.isolated

    def is_goal(self):
        """Check if the current state is the goal state."""
        return self.board == self.goal

    def move(self, x1, y1, x2, y2):
        """Move a peg from (x1, y1) to (x2, y2)."""
        if not self.is_cell_valid(x1, y1) or not self.is_cell_valid(x2, y2):
            return None
        else:
            c1 = y1 * self.x_size + x1
            c2 = y2 * self.x_size + x2
            tmp = self.board[c2]
            self.board[c2] = self.board[c1]
            self.board[c1] = 2
            return tmp

    def remove(self, x, y):
        """Remove a peg at position (x, y)."""
        if not self.is_cell_valid(x, y):
            return None
        else:
            c = y * self.x_size + x
            tmp = self.board[c]
            self.board[c] = 2
            self.pegs -= 1
            return tmp

    def is_cell_valid(self, x, y):
        """Check if the cell (x, y) is a valid position on the board."""
        return (x < self.x_size and x >= 0) and (y < self.y_size and y >= 0)

    def get_isolated(self):
        """Count isolated pegs on the board."""
        return self.isolated

    def cost(self, state):
        """Return the cost of moving from the current state to a child state."""
        return 1

    def __str__(self):
        """String representation of the game board."""
        out = ''
        for i in range (0, len(self.board)):
            s = ''
            if i % self.x_size == 0 and i != 0:
                out += '\n'
            if self.board[i] == 2:
                s += 'O'
            elif self.board[i] == 0:
                s += '-'
            elif self.board[i] == 1:
                s += '*'
            out += s
        return out

    def __eq__(self, other):
        """Check if two game states are equal."""
        return self.board == other.board

    def __hash__(self):
        """Return the hash value of the game state."""
        return hash(tuple(self.board))

# Function to count pegs on the board
def count_pegs(board):
    """Count the number of pegs on the board."""
    count = 0
    for peg in board:
        if peg == 1:
            count += 1
    return count

# Function to get successors of a given state
def get_successors(state):
    """Generate all possible successor states of a given state."""
    children, moves = [], []
    count = 0
    isolated = 0
    for cell in range(0, len(state.board)):
        if state.board[cell] == 1:
            move_n = move_north(state, cell)
            move_s = move_south(state, cell)
            move_w = move_west(state, cell)
            move_e = move_east(state, cell)
            moves.append(move_n)
            moves.append(move_s)
            moves.append(move_w)
            moves.append(move_e)
            if move_n is None and move_s is None and move_e is None and move_w is None:
                isolated += 1
    for move in moves:
        if move:
            children.append(move)
            count += 1
    state.children = count
    state.isolated = isolated
    return children

# Functions to perform moves in different directions
def move_north(state, cell):
    """Move a peg northward."""
    new_state = GameState(list(state.board), state.goal, state.x_size, state.y_size, state.pegs)
    y = int(cell / state.x_size)
    x = cell % state.x_size

    if not new_state.is_cell_valid(x, y - 2):
        return None
    else:
        new_cell = (y - 2) * state.x_size + x
        btw_cell = (y - 1) * state.x_size + x
        if new_state.board[new_cell] != 2 or new_state.board[btw_cell] != 1:
            return None
        else:
            new_state.remove(x, y - 1)
            new_state.move(x, y, x, y - 2)
            new_state.parent = state
            return new_state

# Define move_south, move_east, and move_west functions similarly

# Define the Breadth-First Search algorithm
def bfs(problem):
    """Perform Breadth-First Search to find the solution."""
    queue, closed = [], set()
    queue.append(problem)
    while queue:
        state = queue.pop(0)
        if state not in closed:
            closed.add(state)
            if state.is_goal():
                return state
            for new_state in get_successors(state):
                queue.append(new_state)
    return None

# Define the Depth-First Search algorithm
def dfs(problem):
    """Perform Depth-First Search to find the solution."""
    stack, closed = [], set()
    stack.append(problem)
    while stack:
        state = stack.pop()
        if state not in closed:
            closed.add(state)
            if state.is_goal():
                return state
            for new_state in get_successors(state):
                stack.append(new_state)
    return None

# Define the main function
def main():
    """Run the different search algorithms and print the solutions."""
    bfs_solution()
    dfs_solution()

# Call the main function if the script is executed directly
if __name__ == "__main__":
    main()

