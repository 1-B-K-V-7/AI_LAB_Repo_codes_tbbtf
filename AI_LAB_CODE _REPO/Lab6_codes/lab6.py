import numpy as np
import re

# Problem 1
book = 'war_and_peace.txt'
file = open(book, 'r', encoding='utf-8')
text = file.read()
file.close()

# Removing the punctuations and converting to lower case
text = re.sub(r'[^a-zA-Z]', " ", text)
text = " ".join(text.split()).lower()[:100000]

# Creating a dictionary of all the unique characters
char_dict = {}
for i in range(26):
    char_dict[chr(i + 97)] = i
char_dict[" "] = 26

# Initialize the parameters
O = np.zeros(len(text), dtype=int)
for i in range(len(text)):
    O[i] = char_dict[text[i]]

# Initial state distribution
pi = np.array(([0.525483, 0.474517]))

# Observable sequence
B = np.array([[0.03735, 0.03408, 0.03455, 0.03828, 0.03782, 0.03922, 0.03688, 
               0.03408, 0.03875, 0.04062, 0.03735, 0.03968, 0.03548, 0.03735, 0.04062, 
               0.03595, 0.03641, 0.03408, 0.04062, 0.03548, 0.03922, 0.04062, 0.03455, 
               0.03595, 0.03408, 0.03408, 0.03688],
              [0.03909, 0.03537, 0.03537, 0.03909, 0.03583, 0.03630, 0.04048, 
               0.03537, 0.03816, 0.03909, 0.03490, 0.03723, 0.03537, 0.03909, 0.03397, 
               0.03397, 0.03816, 0.03676, 0.04048, 0.03443, 0.03537, 0.03955, 
               0.03816, 0.03723, 0.03769, 0.03955, 0.03397]])

# Transition matrix
A = np.array([[0.47468, 0.52532], [0.51656, 0.48344]])

# Set of possible observations
V = np.array(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 
              'u', 'v', 'w', 'x', 'y', 'z', ' '])

# Set of possible states, Q is hidden
# Number of observation symbols
M = len(V)
# Number of states in the model
N = len(A)
# Length of observation sequence
T = len(O)

# Alpha Pass
def alpha_pass(A1, B1, pi1, O1):
    c1 = np.zeros([T, 1])
    alpha1 = np.zeros([T, N])
    c1[0][0] = 0
    for x in range(N):
        alpha1[0][x] = pi1[x] * B1[x][O1[0]]
        c1[0][0] = c1[0][0] + alpha1[0][x]
    c1[0][0] = 1/c1[0][0]
    for x in range(N):
        alpha1[0][x] = c1[0][0] * alpha1[0][x]
    for t in range(1, T):
        c1[t][0] = 0
        for x in range(N):
            alpha1[t][x] = 0
            for y in range(N):
                alpha1[t][x] = alpha1[t][x] + alpha1[t-1][y] * A1[y][x]
            alpha1[t][x] = alpha1[t][x] * B1[x][O1[t]]
            c1[t][0] = c1[t][0] + alpha1[t][x]
        c1[t][0] = 1/c1[t][0]
        for x in range(N):
            alpha1[t][x] = c1[t][0] * alpha1[t][x]
    return alpha1, c1

# Beta Pass
def beta_pass(A1, B1, O1, c1):
    beta1 = np.zeros([T, N])
    for x in range(N):
        beta1[T-1][x] = c1[T-1][0]
    for t in range(T-2, -1, -1):
        for x in range(N):
            beta1[t][x] = 0
            for y in range(N):
                beta1[t][x] = beta1[t][x] + A1[x][y] * B1[y][O1[t + 1]] * beta1[t + 1][y]
            beta1[t][x] = c1[t][0] * beta1[t][x]
    return beta1

# Compute Gamma(x,t) and Gamma(x,y,t)
def gamma_pass(alpha1, beta1, A1, B1, O1):
    gamma1 = np.zeros([T, N])
    di_gamma1 = np.zeros([T, N, N])
    for t in range(T-1):
        for x in range(N):
            gamma1[t][x] = 0
            for y in range(N):
                di_gamma1[t][x][y] = alpha1[t][x] * A1[x][y] * B1[y][O1[t + 1]] * beta1[t + 1][y]
                gamma1[t][x] = gamma1[t][x] + di_gamma1[t][x][y]
    for x in range(N):
        gamma1[T-1][x] = alpha1[T-1][x]
    return gamma1, di_gamma1

# Re-estimate A, B, pi
def re_estimate(gamma1, di_gamma1, A1, B1, pi1):
    for x in range(N):
        pi1[x] = gamma1[0][x]
    for x in range(N):
        denominator = 0
        for t in range(T-1):
            denominator = denominator + gamma1[t][x]
        for y in range(N):
            numerator = 0
            for t in range(T-1):
                numerator = numerator + di_gamma1[t][x][y]
            A1[x][y] = numerator/denominator
    for x in range(N):
        denominator = 0
        for t in range(T):
            denominator = denominator + gamma1[t][x]
        for y in range(M):
            numerator = 0
            for t in range(T):
                if O[t] == y:
                    numerator = numerator + gamma1[t][x]
            B1[x][y] = numerator/denominator
    return A1, B1, pi1

# Compute log[P(O|lambda)]
def log_prob(c1):
    logProb1 = 0
    for x in range(T):
        logProb1 = logProb1 + np.log(c1[x][0])
    logProb1 = -logProb1
    return logProb1

# Values initially
oldLogProb = -10000000
print("A: \n", A)
print("B: \n", np.concatenate((V.reshape(1, M), B), axis=0).T)
print("pi: ", pi)
print("logProb: ", oldLogProb)

# After 100 iterations
maxIter = 100
for ite in range(maxIter):
    alpha, c = alpha_pass(A, B, pi, O)
    beta = beta_pass(A, B, O, c)
    gamma, di_gamma = gamma_pass(alpha, beta, A, B, O)
    A, B, pi = re_estimate(gamma, di_gamma, A, B, pi)
    logProb = log_prob(c)
    print("A: \n", A)
    print("B: \n", np.concatenate((V.reshape(1, M), np.round_(B, decimals=7)), axis=0).T)
    print("pi: ", np.round_(pi, decimals=5))
    print("logProb: ", logProb)

# Problem 2
import pandas as pd
import numpy as np
from scipy.special import comb

df = pd.read_csv('/content/2020_ten_bent_coins.csv').transpose()

# O being tail and 1 being head
# counting number of heads and tails
np.random.seed(0)
heads = df.sum().to_numpy() #numpy array
tails = 100 - heads
selected_coin = np.random.randint(0,10,size=(500,)) 
#creating an array of 500 values with each one having 
value ranging from 1 to 10
_, count_selected_coin = np.unique(selected_coin,return_counts = True) # count of 
#which coin has been selected how many times
MLE_vector = np.zeros(10) #maximum likelihood estimation

for i,j in zip(heads, selected_coin):
    MLE_vector[j] += i

# The MLE vector is then divided by the product of the 
# count of the selected coin and the total number
# of tosses (100) to obtain the MLE estimates of the 
# unknown bias values.
MLE_vector = MLE_vector/(count_selected_coin*100)

# A function compute_likelihood is defined to calculate the
# likelihood of a given observation (number of heads) given 
# the number of tosses and the estimated bias value.
def compute_likelihood(obs, n, pheads):
    likelihood = comb(n, obs, exact=True)*(pheads**obs)*((1.0-pheads)**(n-obs))
    return likelihood

# The MLE estimates are updated using the Expectation-Maximization (EM) algorithm. 
# In each iteration (or epoch) of the EM algorithm,
# the expected values of the number of heads and tails for each coin are calculated 
# based on the current MLE estimates.
# The MLE estimates are then updated based on these expected values.
np.random.seed(0)
p_heads = np.zeros((100,10))
p_heads[0]=np.random.random((1,10))
print(p_heads[0])

# The loop continues until the improvement in the MLE estimates between two 
# consecutive iterations is less than a threshold eps, which is set to 0.01.
eps = 0.01
improvement = float('inf') #positive infinity
epoch = 0
while improvement>eps:
    expectation = np.zeros((10,500,2))
    for i in range(500):
        e_head = heads[i]
        e_tail = tails[i]
        likelihood = np.zeros(10)
        for j in range(10):
            likelihood[j]=compute_likelihood(e_head,100,p_heads[epoch][j])
        weights = likelihood/np.sum(likelihood)
        for j in range(10):
            expectation[j][i] = weights[j]*np.array([e_head,e_tail])
    theta = np.zeros(10)
    for i in range(10):
        theta[i] = np.sum(expectation[i],axis=0)[0]/np.sum(expectation[i])
    p_heads[epoch+1] = theta
    print(f'Epoch ->{epoch}\n Theta ->{theta}')
    improvement = max(abs(p_heads[epoch+1]-p_heads[epoch]))
    epoch+=1

# The MLE estimates are stored in the theta variable, which is the final output of the code.
for i, j in enumerate(theta): # to get the index as well 
    print(f"{i+1} : {j:.3f}")

#problem 3
import pandas as pd  # Data manipulation.
import numpy as np  # Numerical operations.
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

# Reading the data from csv file
data1 = pd.read_csv("2020_em_clustering.csv", sep=',', header=None)
data1 = data1.transpose()

# Defining the number of clusters for KMeans
kmeans_model = KMeans(n_clusters=2)

# We use the kmeans_model to predict the cluster labels for each data point.
kmeans_model.fit(data1)
kmeans_predictions = kmeans_model.predict(data1)

# Plotting the clusters for KMeans
plt.scatter(data1.iloc[:, 0], [i for i in range(data1.shape[0])], c=kmeans_predictions)
plt.xlabel("position")
plt.ylabel("Classification")
plt.show()

# Reading the data from csv file
data2 = pd.read_csv("2020_em_clustering.csv", sep=',', header=None)
data2 = data2.transpose()

# Defining the number of clusters for Gaussian Mixture Model (EM)
em_model = GaussianMixture(n_components=2)

# Fitting the EM model
"""
Fitting the model: The fit method is used to fit the EM algorithm to the data. 
This step involves iteratively estimating the parameters of the Gaussian 
distribution for each cluster until convergence.
"""
em_model.fit(data2)

"""
Predicting the clusters: The predict method is used to make predictions on 
which cluster each data point belongs to.
"""
em_predictions = em_model.predict(data2)

# Plotting the clusters for Gaussian Mixture Model (EM)
plt.scatter(data2.iloc[:, 0], [i for i in range(data2.shape[0])], c=em_predictions)
plt.xlabel("position")
plt.ylabel("Classification")
plt.show()
