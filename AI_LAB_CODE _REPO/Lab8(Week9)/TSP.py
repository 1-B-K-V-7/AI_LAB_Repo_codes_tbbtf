# # Using Gini Index as the Splitting Criterion
# ## Single Decision Tree
# %%
# Initialize and train Decision Tree Classifier
clf_single = DecisionTreeClassifier(criterion="gini")
clf_single.fit(X_train, y_train)
# Display the tree plot
plot_tree(clf_single)[-1]
# %%
# Make predictions and evaluate performance
predictions_single = clf_single.predict(X_test)
accuracy_single = accuracy_score(y_test, predictions_single)
f1score_single = f1_score(y_test, predictions_single, average='weighted')
print("Accuracy (Single Tree): ", accuracy_single)
print("F1 score (Single Tree): ", f1score_single)
# %% [markdown]
# ## Ensemble of 20 Decision Trees
# %%
# Lists to store accuracies and F1 scores
accuracies_ensemble = []
f_scores_ensemble = []
# Number of iterations
num_iterations = 20
for i in range(num_iterations):
    # Split the data
    X_train_i, X_test_i, y_train_i, y_test_i = train_test_split(encoded_data, target_encoded, test_size=0.4, random_state=i)
    # Initialize and train Decision Tree Classifier
    clf_ensemble = DecisionTreeClassifier(criterion='gini')
    clf_ensemble.fit(X_train_i, y_train_i)
    # Make predictions
    y_pred_ensemble = clf_ensemble.predict(X_test_i)
    # Calculate F1-score
    f1_ensemble = f1_score(y_test_i, y_pred_ensemble, average='weighted')
    # Calculate accuracy from confusion matrix
    accuracy_ensemble = accuracy_score(y_test_i, y_pred_ensemble)
    # Append accuracy and F1-score to lists
    accuracies_ensemble.append(accuracy_ensemble)
    f_scores_ensemble.append(f1_ensemble)
# Calculate average accuracy and F1-score
avg_accuracy_ensemble = np.mean(accuracies_ensemble)
avg_f1_score_ensemble = np.mean(f_scores_ensemble)
print("Average Accuracy (Ensemble):", avg_accuracy_ensemble)
print("Average F1-Score (Ensemble):", avg_f1_score_ensemble)
