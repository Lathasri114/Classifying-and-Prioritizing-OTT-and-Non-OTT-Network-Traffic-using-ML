import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load your dataset, replace 'data.csv' with your dataset file path
data = pd.read_csv('modified_dataset.csv')

# Define the number of rows to randomly select
n_rows = 1000  # Change this to your desired number of rows

# Randomly select a subset of rows
data = data.sample(n=n_rows, random_state=42)

# Replace 'label' with the actual name of your target column
target_column_name = 'label'

# Split the data into features (X) and target (y)
X = data.drop(target_column_name, axis=1)
y = data[target_column_name]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define a range of hyperparameters to search over
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],  # Regularization strength
    'penalty': ['l1', 'l2'],  # Penalty type
    'solver': ['liblinear', 'saga']  # Solver algorithm
}

# Initialize an empty list to store accuracy scores
accuracy_scores = []

# Iterate over hyperparameter combinations
for C in param_grid['C']:
    for penalty in param_grid['penalty']:
        for solver in param_grid['solver']:
            # Create and train the Logistic Regression model with specific hyperparameters
            model = LogisticRegression(C=C, penalty=penalty, solver=solver, max_iter=1000, random_state=42)
            model.fit(X_train_scaled, y_train)

            # Make predictions on the test data
            y_pred = model.predict(X_test_scaled)

            # Calculate accuracy and append to the list
            accuracy = accuracy_score(y_test, y_pred)
            accuracy_scores.append((C, penalty, solver, accuracy))

# Separate the accuracy scores by hyperparameters for plotting
accuracy_scores = np.array(accuracy_scores)
C_values = accuracy_scores[:, 0]
penalty_values = accuracy_scores[:, 1]
solver_values = accuracy_scores[:, 2]
accuracy_values = accuracy_scores[:, 3]

# Plot the accuracy graph
plt.figure(figsize=(12, 8))
plt.scatter(C_values, accuracy_values, c=penalty_values, cmap='viridis', marker='o')
plt.colorbar(label='penalty')
plt.xlabel('C (Regularization strength)')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. C and Penalty for Logistic Regression')
plt.grid()
plt.show()
