import math
import random
import matplotlib.pyplot as plt
import animated_visualizer
from graph import Graphing

class SimulatedAnnealing:
    def __init__(self, coordinates, place, stopping_iter):
        self.coordinates = coordinates
        self.place = place
        self.N = len(coordinates)
        self.stopping_temperature = 1e-8
        self.temperature = 1000
        self.stopping_iter = stopping_iter
        self.iteration = 1
        self.nodes = [i for i in range(self.N)]
        self.best_path = None
        self.best_cost = float("inf")
        self.cost_list = []
        self.path_history = []

    def path_cost(self, solution):
        cost = 0
        for i in range(self.N):
            cost += self.distance(solution[i % self.N], solution[(i + 1) % self.N])
        return cost

    def distance(self, node0, node1):
        coord0, coord1 = self.coordinates[node0], self.coordinates[node1]
        distance = math.sqrt(((coord0[0] - coord1[0]) ** 2) + ((coord0[1] - coord1[1]) ** 2))
        return distance

    def accept(self, candidate):
        candidate_cost = self.path_cost(candidate)
        if candidate_cost < self.best_cost:
            self.best_cost = candidate_cost
            self.best_path = candidate
        else:
            probability_accept = math.exp(-abs(candidate_cost - self.current_cost) / self.temperature)
            if random.random() < probability_accept:
                self.current_cost = candidate_cost
                self.current_path = candidate

    def initial_solution(self):
        current_node = random.choice(self.nodes)
        path = [current_node]
        remaining_nodes = set(self.nodes)
        remaining_nodes.remove(current_node)
        while remaining_nodes:
            next_node = min(remaining_nodes, key=lambda x: self.distance(current_node, x))
            current_node = next_node
            remaining_nodes.remove(current_node)
            path.append(current_node)
        initial_cost = self.path_cost(path)
        if self.best_cost > initial_cost:
            self.best_cost = initial_cost
            self.best_path = path
        self.cost_list.append(initial_cost)
        self.path_history.append(path)
        return path, initial_cost

    def simulated_annealing(self):
        self.current_path, self.current_cost = self.initial_solution()
        while self.temperature >= self.stopping_temperature and self.iteration < self.stopping_iter:
            candidate = list(self.current_path)
            l = random.randint(2, self.N - 1)
            i = random.randint(0, self.N - 1)
            candidate[i:(i+l)] = reversed(candidate[i:(i+l)])
            self.accept(candidate)
            self.temperature *= 0.9995
            self.iteration += 1
            self.cost_list.append(self.current_cost)
            self.path_history.append(self.current_path)
        print("Best cost obtained:", self.best_cost)

    def display_optimal_path(self):
        n = len(self.best_path)
        tour = ''
        for i in range(n):
            x = self.best_path[i]
            tour = tour + self.place[x]+' -> '
        tour += self.place[self.best_path[0]]
        print("Optimal Path:", tour)
        graph = Graphing(self.coordinates, self.place, self.best_path)
        graph.plot_graph()

    def animate_solutions(self):
        animated_visualizer.animateTSP(self.path_history, self.coordinates, self.best_path, self.place)

    def plot_learning(self):
        initial_cost = self.cost_list[0]
        plt.plot([i for i in range(len(self.cost_list))], self.cost_list)
        line_init = plt.axhline(y=initial_cost, color='r', linestyle='--')
        line_min = plt.axhline(y=self.best_cost, color='g', linestyle='-.')
        plt.title("Learning Progress")
        plt.legend([line_init, line_min], ['Initial Cost', 'Optimized Cost'])
        plt.ylabel("Cost")
        plt.xlabel("Iteration")
        plt.show()
