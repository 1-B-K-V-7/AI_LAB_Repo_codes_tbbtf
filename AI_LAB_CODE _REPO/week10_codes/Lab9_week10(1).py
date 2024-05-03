import random
import numpy as np
import matplotlib.pyplot as plt

random.seed(10)


class BinaryBandit(object):
    def __init__(self):
        # N = number of arms
        self.N = 2

    def actions(self):
        # Return possible actions
        return list(range(self.N))

    def reward1(self, action):
        # Reward function for the first bandit
        p = [0.1, 0.2]  # Winning probabilities for each action
        rand = random.random()
        if rand < p[action]:
            return 1  # Win
        else:
            return 0  # Lose

    def reward2(self, action):
        # Reward function for the second bandit
        p = [0.8, 0.9]  # Winning probabilities for each action
        rand = random.random()
        if rand < p[action]:
            return 1  # Win
        else:
            return 0  # Lose


def eGreedy_binary_r1(myBandit, epsilon, max_iteration):

    Q = [0] * myBandit.N  # Estimated values for each action
    count = [0] * myBandit.N  # Number of times each action is taken
    R = []  # List to store rewards per iteration
    R_avg = [0] * 1  # List to store average rewards
    max_iter = max_iteration

    # Incremental Implementation
    for iter in range(1, max_iter):
        if random.random() > epsilon:
            action = Q.index(max(Q))  # Exploit/Greed
        else:
            action = random.choice(myBandit.actions())  # Explore
        r = myBandit.reward1(action)  # Get reward for the chosen action
        R.append(r)
        count[action] += 1
        Q[action] += (r - Q[action]) / count[action]  # Update action value estimate
        R_avg.append(R_avg[iter - 1] + (r - R_avg[iter - 1]) / iter)  # Update average reward
    return Q, R_avg, R


def eGreedy_binary_r2(myBandit, epsilon, max_iteration):
    # Epsilon-Greedy algorithm for the second bandit
    # Initialization
    Q = [0] * myBandit.N  # Estimated values for each action
    count = [0] * myBandit.N  # Number of times each action is taken
    R = []  # List to store rewards per iteration
    R_avg = [0] * 1  # List to store average rewards
    max_iter = max_iteration

    # Incremental Implementation
    for iter in range(1, max_iter):
        if random.random() > epsilon:
            action = Q.index(max(Q))  # Exploit/Greed
        else:
            action = random.choice(myBandit.actions())  # Explore
        r = myBandit.reward2(action)  # Get reward for the chosen action
        R.append(r)
        count[action] += 1
        Q[action] += (r - Q[action]) / count[action]  # Update action value estimate
        R_avg.append(R_avg[iter - 1] + (r - R_avg[iter - 1]) / iter)  # Update average reward
    return Q, R_avg, R


# Seed for reproducibility
random.seed(10)
# Instantiate BinaryBandit object
myBandit = BinaryBandit()

# Run Epsilon-Greedy algorithm for the first bandit
Q, R_avg, R = eGreedy_binary_r1(myBandit, 0.2, 2000)



