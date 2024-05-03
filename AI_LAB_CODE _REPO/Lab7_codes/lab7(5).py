import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.preprocessing import OneHotEncoder

# Load the dataset
columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
data = pd.read_csv("car.data", names=columns)

# Map class labels to numerical values
class_mapping = {'unacc': 0, 'acc': 1, 'good': 2, 'vgood': 3}
data['class'] = data['class'].map(class_mapping)

# Convert categorical features into one-hot encoded representation
categorical_columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']
data_encoded = pd.get_dummies(data, columns=categorical_columns)

# Split the dataset into features and target
X = data_encoded.drop('class', axis=1)
y = data_encoded['class']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, stratify=y)

# Initialize the decision tree classifier
clf_ent = DecisionTreeClassifier(criterion='entropy')

# Train the classifier on the training data
clf_ent.fit(X_train, y_train)

# Test the classifier on the testing data
y_pred_test = clf_ent.predict(X_test)

# Evaluate the classifier on the testing data
conf_matrix_test = confusion_matrix(y_test, y_pred_test)
accuracy_test = accuracy_score(y_test, y_pred_test)
f1_test = f1_score(y_test, y_pred_test, average="weighted")

# Test the classifier on the training data
y_pred_train = clf_ent.predict(X_train)

# Evaluate the classifier on the training data
conf_matrix_train = confusion_matrix(y_train, y_pred_train)
accuracy_train = accuracy_score(y_train, y_pred_train)
f1_train = f1_score(y_train, y_pred_train, average="weighted")

# Print results for the testing data
print("Testing Data:")
print("Confusion Matrix:")
print(conf_matrix_test)
print("Accuracy:", accuracy_test)
print("F1 Score:", f1_test)

# Print results for the training data
print("\nTraining Data:")
print("Confusion Matrix:")
print(conf_matrix_train)
print("Accuracy:", accuracy_train)
print("F1 Score:", f1_train)
