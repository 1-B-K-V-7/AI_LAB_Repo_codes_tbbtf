%------------------------------------------------------------
% Hopfield Network Example
% Demonstration of pattern storage
% CS308 Introduction to Artificial Intelligence
% Author: [Your Name]
% Date: [Current Date]
% Place: [Your Institution]
% Ref: Information, Inference and Learning Algorithms, D McKay
%------------------------------------------------------------

clear all; % Clear all variables from memory
close all; % Close all figures
clc; % Clear the command window

%--------------------------------------------
% Define Input Patterns
% D, J, C, M
%--------------------------------------------
patterns = [1 1 1 1 -1 -1 1 -1 -1 1 -1 1 -1 -1 1 -1 1 -1 -1 1 -1 1 1 1 -1;
            1 1 1 1 1 -1 -1 -1 1 -1 -1 -1 -1 1 -1 1 -1 -1 1 -1 1 1 1 -1 -1;
           -1 1 1 1 1 1 -1 -1 -1 -1 1 -1 -1 -1 -1 1 -1 -1 -1 -1 -1 1 1 1 1;
            1 -1 -1 -1 1 1 1 -1 1 1 1 -1 1 -1 1 1 -1 -1 -1 1 1 -1 -1 -1 1]';

%--------------------------------------------
% Learn the weights according to Hebb's rule
%--------------------------------------------
[num_patterns, pattern_length] = size(patterns); % Get the dimensions of input patterns (25 x 4)
weights = zeros(pattern_length, pattern_length); % Initialize weight matrix (25 x 25)
for i = 1:num_patterns % For each pattern
    weights = weights + patterns(:,i)*patterns(:,i)'; % Update weights using Hebb's rule
end
weights(logical(eye(size(weights)))) = 0; % Set diagonal elements to zero (no self-connections)
weights = weights / num_patterns; % Normalize weights

selected_pattern = patterns(:,1); % Select a pattern (D => 1; J => 2; C => 3; M => 4;)
min_error = pattern_length; % Initialize minimum error
avg_error = 0; % Initialize average error

% Perform flipping and retrieval
for j = 1:1000 % Repeat the retrieval process 1000 times
    for num_flipped_bits = 1:pattern_length % For each possible number of flipped bits
        pattern_copy = selected_pattern; % Create a copy of the pattern
        indices_to_flip = randperm(length(pattern_copy), num_flipped_bits); % Randomly select indices to flip
        pattern_copy(indices_to_flip) = -pattern_copy(indices_to_flip); % Flip selected bits
        output = sign(weights * pattern_copy); % Calculate output using the weight matrix
        error = norm(output - selected_pattern); % Calculate error between output and original pattern
        max_iterations = 25; % Maximum iterations for convergence
        while error > 1 && max_iterations > 0 % While error is greater than 1 and maximum iterations are not reached
            output = sign(weights * output); % Update output using the weight matrix
            error = norm(output - selected_pattern); % Calculate new error
            max_iterations = max_iterations - 1; % Decrement iteration counter
        end
        if error > 1 % If error is still greater than 1 after convergence
            imshow(reshape(-output, 5, 5)'); % Display the retrieved pattern
            min_error = min(num_flipped_bits, min_error); % Update minimum error
            avg_error = avg_error + num_flipped_bits; % Update average error
            break % Break the loop
        end
    end
end

disp(min_error); % Display minimum error
disp(avg_error / 1000); % Display average error
