import matplotlib.pyplot as plt

class Graphing:
    def __init__(self, coordinates, place, best_path):
        self.coordinates = coordinates
        self.place = place
        self.best_path = best_path

    def plot_graph(self):
        num_points = len(self.best_path)
        x = [0] * (num_points + 1)
        y = [0] * (num_points + 1)
        for i in range(num_points):
            x[i] = self.coordinates[self.best_path[i]][0]
            y[i] = self.coordinates[self.best_path[i]][1]
        x[num_points] = self.coordinates[self.best_path[0]][0]
        y[num_points] = self.coordinates[self.best_path[0]][1]
        plt.plot(x, y, color='green', linestyle='dashed', linewidth=1, marker='o', markerfacecolor='blue', markersize=2)
        annotations = [""] * (num_points + 1)
        for i in range(num_points):
            index = self.best_path[i]
            annotations[i] = self.place[index]
        annotations[num_points] = self.place[self.best_path[0]]
        for i, label in enumerate(annotations):
            plt.annotate(label, (x[i], y[i]))
        plt.show()
