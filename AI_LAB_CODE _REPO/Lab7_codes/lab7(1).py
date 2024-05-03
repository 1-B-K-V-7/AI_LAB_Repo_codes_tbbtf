import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
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

# Split data into features and target
X = data_encoded.drop('class', axis=1)
y = data_encoded['class']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, stratify=y)

# Initialize decision tree classifier
clf_ent = DecisionTreeClassifier(criterion='entropy')

# Train the classifier
clf_ent.fit(X_train, y_train)

# Test the classifier
y_pred = clf_ent.predict(X_test)

# Evaluate the classifier
conf_matrix = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="weighted")

print("Confusion Matrix:")
print(conf_matrix)
print("\nF1 Score:", f1)
print("\nAccuracy:", accuracy)
