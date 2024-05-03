import numpy as np
import scipy.io

file_path = r"C:\Users\Himanshu\Desktop\scrambled.mat"
f = open(file_path)
for _ in range(5):
    f.readline()

data = []
line = f.readline()
while line[1:] != "":
    val = int(line[1:])
    data.append(val)
    line = f.readline()

input_data = np.array(data)
input_matrix = input_data.reshape(512, 512)
input_matrix = input_matrix.T

import matplotlib.pyplot as plt

plt.imshow(input_matrix)
plt.show()


class Energy:
    def __init__(self, image):
        self.image = image
        self.height = 4
        self.width = 4

    def getLeftRightEnergy(self, tile):
        try:
            i, j = tile
            x1 = 128 * i
            x2 = 128 * (i + 1)
            y = 128 * (j + 1) - 1
            diff = self.image[x1:x2, y] - self.image[x1:x2, y + 1]
            return np.sqrt((diff**2).mean())
        except IndexError:
            return 0

    def getUpDownEnergy(self, tile):
        try:
            i, j = tile
            y1 = 128 * j
            y2 = 128 * (j + 1)
            x = 128 * (i + 1) - 1
            diff = self.image[x, y1:y2] - self.image[x + 1, y1:y2]
            return np.sqrt((diff**2).mean())
        except IndexError:
            return 0

    def getEnergyAround(self, tile):
        i, j = tile
        e = np.zeros(4)
        e[0] = self.getLeftRightEnergy((i, j - 1))
        e[1] = self.getLeftRightEnergy((i, j))
        e[2] = self.getUpDownEnergy((i - 1, j))
        e[3] = self.getUpDownEnergy((i, j))
        return e.sum()

    def getEnergyAround2Tiles(self, t1, t2):
        return self.getEnergyAround(t1) + self.getEnergyAround(t2)

    def energy(self):
        energy = 0
        for i in range(1, self.height - 1):
            for j in range(1, self.width - 1):
                energy += self.getEnergyAround((i, j))
        return energy

    def cheatEnergy(self):
        return np.linalg.norm(self.image - original)


e = Energy(input_matrix)
e.energy()

max_iterations = 60000
initial_temperature = 1000
stopping_temperature = 0.00005
temperature_decay = 0.9995

x = np.arange(0, 4)
y = np.arange(0, 4)
current_iteration = 0
e = Energy(image=input_matrix)
best_cost = e.energy()
best = input_matrix
cost_list = []

while current_iteration != max_iterations and initial_temperature >= 0:

    new_image = best.copy()
    np.random.shuffle(x)
    np.random.shuffle(y)

    cost_old = Energy(image=new_image).getEnergyAround2Tiles((x[0], y[0]), (x[1], y[1]))
    new_image[128 * x[0] : 128 * x[0] + 128, 128 * y[0] : 128 * y[0] + 128] = best[
        128 * x[1] : 128 * x[1] + 128, 128 * y[1] : 128 * y[1] + 128
    ]
    new_image[128 * x[1] : 128 * x[1] + 128, 128 * y[1] : 128 * y[1] + 128] = best[
        128 * x[0] : 128 * x[0] + 128, 128 * y[0] : 128 * y[0] + 128
    ]

    cost_new = Energy(image=new_image).getEnergyAround2Tiles((x[0], y[0]), (x[1], y[1]))

    if cost_new < cost_old:
        best = new_image
        best_cost = Energy(image=new_image).energy()

    elif np.random.rand() < np.exp(-abs(cost_old - cost_new) / initial_temperature):
        best = new_image
        best_cost = Energy(image=new_image).energy()

    initial_temperature = initial_temperature * temperature_decay
    current_iteration += 1
    cost_list.append(best_cost)

    if current_iteration == 1 or current_iteration % 20000 == 0:
        print("Iteration no -", current_iteration)
        print("Best Cost -", best_cost)
        plt.imshow(best)
        plt.show()

print("Energy of original figure ", Energy(image=original).energy())
plt.plot(cost_list)
plt.xlabel("Iterations", fontsize=15)
plt.ylabel("diff_energy", fontsize=15)
plt.show()
