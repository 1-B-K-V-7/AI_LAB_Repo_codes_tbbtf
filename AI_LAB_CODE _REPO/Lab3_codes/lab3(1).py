import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def animate_traveling_salesman(history, points, best_path, place):
    key_frames_multiplier = len(history) // 1500

    fig, ax = plt.subplots()

    line, = plt.plot([], [], lw=2)
    
    def plot_path():
        length = len(best_path)
        x = [0] * (length + 1)
        y = [0] * (length + 1)
        for i in range(0, length):
            x[i] = points[best_path[i]][0]
        for i in range(0, length):
            y[i] = points[best_path[i]][1]
        x[length] = points[best_path[0]][0]
        y[length] = points[best_path[0]][1]
        plt.plot(x, y, color='green', linestyle='dashed', linewidth=1, marker='o', markerfacecolor='blue', markersize=2)
        annotations = [""] * (length + 1)
        for i in range(0, length):
            index = best_path[i]
            annotations[i] = place[index]
        annotations[length] = place[best_path[0]]
        for i, label in enumerate(annotations):
            plt.annotate(label, (x[i], y[i]))

    def initialize():
        x = [points[i][0] for i in history[0]]
        y = [points[i][1] for i in history[0]]
        plt.plot(x, y, 'co')

        extra_x = (max(x) - min(x)) * 0.05
        extra_y = (max(y) - min(y)) * 0.05
        ax.set_xlim(min(x) - extra_x, max(x) + extra_x)
        ax.set_ylim(min(y) - extra_y, max(y) + extra_y)

        line.set_data([], [])
        return line,

    def update_frame(frame):
        x = [points[i][0] for i in history[frame] + [history[frame][0]]]
        y = [points[i][1] for i in history[frame] + [history[frame][0]]]
        line.set_data(x, y)
        return line

    animation = FuncAnimation(fig, update_frame, frames=range(0, len(history), key_frames_multiplier), init_func=initialize, interval=3, repeat=False)
    plot_path()
    plt.show()
