import random


class Bandit(object):
    def __init__(self, N):
        # N = number of arms
        self.N = N
        # Initialize expected rewards for each arm
        self.expRewards = [10] * N

    def actions(self):
        # Return possible actions
        return list(range(0, self.N))

    def reward(self, action):
        # Calculate reward for a given action
        result = []
        # Update expected rewards with Gaussian noise
        for i in range(len(self.expRewards)):
            self.expRewards[i] += random.gauss(0, 0.1)
        # Generate reward with additional Gaussian noise
        result = self.expRewards[action] + random.gauss(0, 0.01)
        return result

# Modified Epsilon-Greedy strategy with a learning rate (alpha)
def eGreedy_modified(myBandit, epsilon, max_iteration, alpha):
    # Initialization
    Q = [0] * myBandit.N  
    count = [0] * myBandit.N  
    r = 0  # Reward
    R = []  # List to store rewards per iteration
    R_avg = [0] * 1 
    max_iter = max_iteration

    # Incremental Implementation
    for iter in range(1, max_iter):
        if random.random() > epsilon:
            action = Q.index(max(Q))  # Exploit/Greed
        else:
            action = random.choice(myBandit.actions())  # Explore
        r = myBandit.reward(action)  # Get reward for the chosen action
        R.append(r)
        count[action] += 1
        # Update action value estimate with a learning rate (alpha)
        Q[action] += alpha * (r - Q[action])
        # Update average reward
        R_avg.append(R_avg[iter - 1] + (r - R_avg[iter - 1]) / iter)
    return Q, R_avg, R
