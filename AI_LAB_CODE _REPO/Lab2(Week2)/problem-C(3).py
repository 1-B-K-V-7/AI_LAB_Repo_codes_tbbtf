# Importing necessary libraries
from string import ascii_lowercase
import random
from itertools import combinations
import numpy as np

# Taking user input for problem parameters
print("Enter the number of clauses ")
num_clauses = int(input())
print("Enter the number of variables in a clause ")
num_variables_in_clause = int(input())
print("Enter number of variables ")
num_variables = int(input())

# Function to create a SAT problem
def create_sat_problem(num_clauses, num_variables_in_clause, num_variables):
    # Lower Case for positive variables, Upper Case for negative variables
    positive_var = (list(ascii_lowercase))[:num_variables]
    negative_var = [c.upper() for c in positive_var]
    variables = positive_var + negative_var
    threshold = 10  # Arbitrary threshold to avoid an infinite loop
    problems = []
    all_combinations = list(combinations(variables, num_variables_in_clause))
    i = 0

    while i < threshold:
        clause = random.sample(all_combinations, num_clauses)
        if clause not in problems:
            i += 1
            problems.append(list(clause))
    return variables, problems

# Creating SAT problem
variables, problems = create_sat_problem(num_clauses, num_variables_in_clause, num_variables)

# Function to randomly assign values to variables
def random_variable_assignment(variables, num_variables):
    # Randomly assigning 0 or 1 to variables
    variable_assignment = list(np.random.choice(2, num_variables))
    negated_assignment = [abs(1 - i) for i in variable_assignment]
    assignment = variable_assignment + negated_assignment
    var_assign = dict(zip(variables, assignment))
    return var_assign

# Function to solve the SAT problem
def solve_sat_problem(problem, assignment):
    count = 0
    for sub in problem:
        values = [assignment[val] for val in sub]
        count += any(values)
    return count

# Hill climbing algorithm to solve SAT problem
def hill_climbing(problem, assignment, parent_num, received, step):
    best_assignment = assignment.copy()      
    assign_values = list(assignment.values())
    assign_keys = list(assignment.keys())
    
    max_num = parent_num
    max_assignment = assignment.copy()
    edit_assignment = assignment.copy()
    
    for i in range(len(assign_values)):
        step += 1
        edit_assignment[assign_keys[i]] = abs(assign_values[i] - 1)
        c = solve_sat_problem(problem, edit_assignment)
        if max_num < c:
            received = step
            max_num = c
            max_assignment = edit_assignment.copy()
            
    if max_num == parent_num:
        s = str(received) + "/" + str(step - len(assign_values))
        return best_assignment, max_num, s
    else:
        parent_num = max_num
        best_assignment = max_assignment.copy()
        return hill_climbing(problem, best_assignment, parent_num, received, step)

# Beam search algorithm to solve SAT problem
def beam_search(problem, assignment, beam_width, step_size):
    best_assignment = assignment.copy()      
    assign_values = list(assignment.values())
    assign_keys = list(assignment.keys())
    steps = []
    possible_assignments = []
    possible_scores = []
    
    edit_assignment = assignment.copy()
    
    initial = solve_sat_problem(problem, assignment)
    if initial == len(problem):
        p = str(step_size) + "/" + str(step_size)
        return assignment, p
    
    for i in range(len(assign_values)):
        step_size += 1
        edit_assignment[assign_keys[i]] = abs(assign_values[i] - 1)
        c = solve_sat_problem(problem, edit_assignment)
        possible_assignments.append(edit_assignment.copy())
        possible_scores.append(c)
        steps.append(step_size)
    
    selected = list(np.argsort(possible_scores))[-beam_width:]
    
    if len(problem) in possible_scores:
        index = [i for i in range(len(possible_scores)) if possible_scores[i] == len(problem)]
        p = str(steps[index[0]]) + "/" + str(steps[-1])
        return possible_assignments[index[0]], p
    else:
        selected_assignments = [possible_assignments[i] for i in selected]
        for a in selected_assignments:
            return beam_search(problem, a, beam_width, step_size)

# Variable neighborhood algorithm to solve SAT problem
def variable_neighbor(problem, assignment, beam_width, step):
    best_assignment = assignment.copy()      
    assign_values = list(assignment.values())
    assign_keys = list(assignment.keys())
    steps = []
    possible_assignments = []
    possible_scores = []
    
    edit_assignment = assignment.copy()
    
    initial = solve_sat_problem(problem, assignment)
    if initial == len(problem):
        p = str(step) + "/" + str(step)
        return assignment, p, beam_width
    
    for i in range(len(assign_values)):
        step += 1
        edit_assignment[assign_keys[i]] = abs(assign_values[i] - 1)
        c = solve_sat_problem(problem, edit_assignment)
        possible_assignments.append(edit_assignment.copy())
        possible_scores.append(c)
        steps.append(step)
    
    selected = list(np.argsort(possible_scores))[-beam_width:]
    
    if len(problem) in possible_scores:
        index = [i for i in range(len(possible_scores)) if possible_scores[i] == len(problem)]
        p = str(steps[index[0]]) + "/" + str(steps[-1])
        return possible_assignments[index[0]], p, beam_width
    
    else:
        selected_assignments = [possible_assignments[i] for i in selected]
        for a in selected_assignments:
            return variable_neighbor(problem, a, beam_width + 1, step)

hill_climbing_assignments = []
assignments = []
hill_num_satisfied = []
initial_satisfied = []
hill_penetration = []
beam_penetration = []
var_penetration = []
var_beam = []
beam_num_satisfied = []
beam_assignments = []
var_assignments = []
i = 0

# Solving each SAT problem using different algorithms
for problem in problems:
    i += 1
    l =[]
    assignment = random_variable_assignment(variables, num_variables)
    initial = solve_sat_problem(problem, assignment)
    best_assignment, score, hp = hill_climbing(problem, assignment, initial, 1, 1)
    hill_climbing_assignments.append(best_assignment)
    assignments.append(assignment)
    hill_num_satisfied.append(score)
    initial_satisfied.append(initial)
    hill_penetration.append(hp)
    
    hill, beam_pen = beam_search(problem, assignment, 3, 1)
    beam_assignments.append(hill)
    beam_penetration.append(beam_pen)
    
    hill, beam_pen = beam_search(problem, assignment, 4, 1)
    
    var, var_pen, bb = variable_neighbor(problem, assignment, 1, 1)
    var_penetration.append(var_pen)
    var_beam.append(bb)
    var_assignments.append(var)
    
    print('Problem ', i, ': ', problem)
    print('HillClimbing: ', best_assignment, ', Penetrance:', hp)
    print('Beam search (3): ', hill, ', Penetrance:', beam_pen)
    print('Beam search (4): ', hill, ', Penetrance:', beam_pen)
    print('Variable Neighborhood: ', var, ', Penetrance:', var_pen)
    print()
