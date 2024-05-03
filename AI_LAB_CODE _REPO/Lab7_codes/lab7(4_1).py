import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder

# Load the data
columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
data = pd.read_csv("car.data", names=columns)

# Map class labels to numerical values
class_mapping = {'unacc': 0, 'acc': 1, 'good': 2, 'vgood': 3}
data['class'] = data['class'].map(class_mapping)

# Convert categorical features into one-hot encoded representation
categorical_columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']
data_encoded = pd.get_dummies(data, columns=categorical_columns)

# Initialize list to store accuracy scores
accuracy_scores = []

# Repeat the exercise 20 times
for i in range(20):
    # Split data into features and target
    X = data_encoded.drop('class', axis=1)
    y = data_encoded['class']
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)
    
    # Initialize decision tree classifier
    clf_ent_20_70 = DecisionTreeClassifier(criterion='entropy')
    
    # Train the classifier
    clf_ent_20_70.fit(X_train, y_train)
    
    # Test the classifier
    y_pred = clf_ent_20_70.predict(X_test)
    
    # Calculate accuracy and store it
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_scores.append(accuracy)
    print(f"Iteration {i+1}: Accuracy = {accuracy}")

# Calculate the average accuracy
average_accuracy = sum(accuracy_scores) / len(accuracy_scores)
print("Average Accuracy of Classification over 20 iterations:", average_accuracy)
