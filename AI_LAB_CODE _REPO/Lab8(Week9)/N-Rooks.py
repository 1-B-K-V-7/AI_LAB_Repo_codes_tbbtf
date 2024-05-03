import numpy as np

# Simplified Energy Equation without using weights
def energy_function(neurons):
    # Initialize energy terms
    energy_row = 0
    energy_col = 0
    
    # Loop over rows
    for i in range(8):
        row_sum = 0
        
        # Calculate row sum
        for j in range(8):
            row_sum += neurons[i][j]
        
        # Accumulate energy based on the sum of neuron activations in each row
        energy_row += (row_sum - 1) ** 2
    
    # Loop over columns
    for i in range(8):
        col_sum = 0
        
        # Calculate column sum
        for j in range(8):
            col_sum += neurons[j][i]
        
        # Accumulate energy based on the sum of neuron activations in each column
        energy_col += (col_sum - 1) ** 2
    
    # Total energy is the sum of row and column energies
    return energy_row + energy_col

# Energy Equation with weights included
def energy_function_with_weights(neurons, weights):
    total_energy = 0
    
    # Loop over neurons
    for i in range(64):
        for j in range(64):
            # Calculate the energy contribution of each neuron pair weighted by corresponding weight
            neuron_i_activation = neurons[i // 8][j % 8]
            neuron_j_activation = neurons[j // 8][i % 8]
            total_energy += neuron_i_activation * neuron_j_activation * weights[j][i]
    
    # Apply the negative factor
    total_energy *= -0.5
    
    # Add penalty term for each neuron being activated
    for i in range(8):
        for j in range(8):
            total_energy += -1 * neurons[i][j]
    
    return total_energy

# Function to print the neuron activations
def print_neuron_activations(neurons):
    for i in range(8):
        for j in range(8):
            if neurons[i][j] == -1:
                print(0, end=" ")  # Print 0 if neuron activation is -1
            else:
                print(neurons[i][j], end=" ")  # Print neuron activation
        print()

# Main function
def main():
    # Initialize neurons with -1 (inactive)
    neurons = [[-1 for _ in range(8)] for _ in range(8)]
    
    # Set the first column of neurons to be activated (value 1)
    for i in range(8):
        neurons[i][0] = 1
    
    # Initialize weights matrix
    weights = [[0 for _ in range(64)] for _ in range(64)]
    
    # Assign weights to connections between neurons
    for i in range(64):
        for j in range(64):
            if i != j:
                if i % 8 == j % 8:
                    weights[i][j] = -2  # Assign -2 if neurons are in the same column
                elif i // 8 == j // 8:
                    weights[i][j] = -2  # Assign -2 if neurons are in the same row
    
    max_iter = 1000  # Maximum number of iterations
    initial_energy = energy_function_with_weights(neurons, weights)  # Initial energy
    
    # Store activated and inactive neurons
    activated_neurons = []
    inactive_neurons = []
    
    for i in range(8):
        for j in range(8):
            if neurons[i][j] == 1:
                activated_neurons.append((i, j))  # Store indices of activated neurons
            else:
                inactive_neurons.append((i, j))  # Store indices of inactive neurons
    
    # Iterate to minimize energy
    for _ in range(max_iter):
        # Randomly choose two neurons, one with value 1 and one with value 0
        x1, y1 = activated_neurons[np.random.randint(0, len(activated_neurons))]
        x2, y2 = inactive_neurons[np.random.randint(0, len(inactive_neurons))]
        
        # Swap activations
        neurons[x1][y1], neurons[x2][y2] = neurons[x2][y2], neurons[x1][y1]
        
        # Compute new energy
        new_energy = energy_function_with_weights(neurons, weights)
        
        if new_energy < initial_energy:
            initial_energy = new_energy  # Update energy if it's reduced
            activated_neurons = []  # Update activated neurons
            inactive_neurons = []  # Update inactive neurons
            
            for i in range(8):
                for j in range(8):
                    if neurons[i][j] == 1:
                        activated_neurons.append((i, j))
                    else:
                        inactive_neurons.append((i, j))
        else:
            # Revert the swap
            neurons[x1][y1], neurons[x2][y2] = neurons[x2][y2], neurons[x1][y1]
    
    # Print final neuron activations
    print_neuron_activations(neurons)

# Run the main function
main()
