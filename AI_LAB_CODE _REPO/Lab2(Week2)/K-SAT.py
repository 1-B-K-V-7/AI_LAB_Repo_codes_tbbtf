# Import necessary libraries
from string import ascii_lowercase
import random
from itertools import combinations

# Function to create a random SAT problem
def generate_sat_problem(num_clauses, num_literals_per_clause, num_variables):
    # Generate lowercase variables for positive literals and corresponding uppercase for negative literals
    positive_literals = (list(ascii_lowercase))[:num_variables]
    negative_literals = [c.upper() for c in positive_literals]
    literals = positive_literals + negative_literals
    
    # Set a threshold to limit the number of attempts to generate distinct problems
    threshold = 10
    problems = []
    all_combinations = list(combinations(literals, num_literals_per_clause))
    attempt_count = 0

    # Loop to generate distinct random problems
    while attempt_count < threshold:
        random_problem = random.sample(all_combinations, num_clauses)
        if random_problem not in problems:
            attempt_count += 1
            problems.append(list(random_problem))
    
    # Convert tuples to lists for each clause
    converted_problems = []
    for clause in problems:
        temp = []
        for literal_combination in clause:
            temp.append(list(literal_combination))
        converted_problems.append(temp)
    return converted_problems

# User input for the number of clauses, number of literals in each clause, and total number of variables
print("Enter the number of clauses ")
num_clauses = int(input())
print("Enter the number of literals in a clause ")
num_literals_per_clause = int(input())
print("Enter the number of variables ")
num_variables = int(input())

# Generate SAT problems
sat_problems = generate_sat_problem(num_clauses, num_literals_per_clause, num_variables)

# Display the generated SAT problems
for i, problem in enumerate(sat_problems):
    print(f"Problem {i + 1}: {problem}")
