"""
File name: cpsc483_midterm_review.py
Purpose: This file contains all computational work
for the first practice exam for CPSC 483 (Intro to
Machine Learning). 
"""

# =============
# Exercise 1
# =============

import numpy as np
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# Load data
iris = datasets.load_iris()
X = iris['data'][:, 3]
y = (iris["target"]==2).astype(int)

# Visualize relationship
plt.scatter(X[y == 0], y[y == 0], color='blue', label='Not Virginica')
plt.scatter(X[y == 1], y[y == 1], color='red', label='Virginica')
plt.xlabel("Petal Width (cm)")
plt.ylabel("Probability")
plt.legend()
plt.show()

# Train logistic regression model
log_reg = LogisticRegression()
log_reg.fit(X.reshape(-1,1), y)

# Create new data points to test: petal widths from 0 to 3 cm
X_new = np.linspace(0, 3, 1000).reshape(-1, 1)
y_proba = log_reg.predict_proba(X_new)
decision_boundary = X_new[y_proba[:, 1] >= 0.5][0]

# Plot the probability curves
plt.plot(X_new, y_proba[:, 1], "g-", linewidth=2, label="Iris-Virginica proba")
plt.plot(X_new, y_proba[:, 0], "b--", linewidth=2, label="Not Iris-Virginica proba")


# Mark the decision boundary with a red dotted line
plt.axvline(x=decision_boundary, color='red', linestyle=':', 
            label=f'Boundary: {decision_boundary[0]:.2f}cm')

plt.xlabel("Petal width (cm)")
plt.ylabel("Probability")
plt.legend(loc="center left")
plt.title("Logistic Regression: Iris-Virginica Classifier")
plt.grid(True, alpha=0.3)
plt.show()

# Find the decision boundary where probability is 50%
print(f"Decision Boundary: {decision_boundary[0]:.2f} cm")
# Typically around 1.6 - 1.7 cm

# 5. Make a manual prediction
test_width = [[1.7]]
prediction = log_reg.predict(test_width)
print(f"Prediction for {test_width[0][0]}cm: {'Iris-Virginica' if prediction[0] == 1 else 'Not Iris-Virginica'}")

# =============
# Exercise 2
# =============

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.linear_model import SGDClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score

# 1. Generate dummy data for demonstration (replace with your actual data)
# X_train (features) and y_train (target labels)
X, y = make_classification(n_samples=200, n_features=20, n_informative=15, n_redundant=5, random_state=42)
X_train = X
y_train = y

# 2. Initialize the scaler and classifier
# Initialize the StandardScaler for feature scaling
scaler = StandardScaler()
# Initialize the SGDClassifier (Stochastic Gradient Descent classifier)
sgd_clf = SGDClassifier(random_state=42)

# 3. Data Preprocessing

# Scale the training features: fit and transform X_train
X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))

# 4. Model Evaluation using cross_val_score
# Perform 3-fold cross-validation to get accuracy scores for each fold
# This gives an estimate of the model's generalization performance
scores = cross_val_score(sgd_clf, X_train_scaled, y_train, cv=3, scoring="accuracy")

# Print the scores and mean accuracy
print(f"Cross-validation scores for each fold: {scores}")
print(f"Mean CV Accuracy: {scores.mean():.4f}")

# 5. Generate Out-of-Sample Predictions using cross_val_predict
# Obtain a prediction for each sample in X_train_scaled using a model trained on other folds
y_train_pred = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=3)

# 6. Evaluate predictions from cross_val_predict
# The predictions can be evaluated against the true labels to compute metrics like accuracy,
# confusion matrix, etc.
overall_accuracy = accuracy_score(y_train, y_train_pred)
print(f"Overall Accuracy from cross_val_predict: {overall_accuracy:.4f}")

# =============
# Exercise 3
# =============

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve

# 1. Define dummy data (Actual labels and predicted probabilities)
# y_true: Actual classes (0 = negative, 1 = positive)
y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1, 1, 0])
# y_scores: Model's estimated probability for the positive class
y_scores = np.array([0.1, 0.4, 0.35, 0.8, 0.2, 0.5, 0.1, 0.7, 0.9, 0.6])

# 2. Calculate the metrics using Scikit-Learn
# Returns arrays of precision/recall values and the corresponding thresholds
precisions, recalls, thresholds_pr = precision_recall_curve(y_true, y_scores)
# Returns False Positive Rate, True Positive Rate, and thresholds for ROC
fpr, tpr, thresholds_roc = roc_curve(y_true, y_scores)

# Function: Plot Precision and Recall against the Threshold
def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    # Plot precision (blue dashed); exclude last value as it has no threshold
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    # Plot recall (green solid); exclude last value
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.legend(loc="center right") 
    plt.grid(True)                 
    plt.xlabel("Threshold")        
    plt.ylabel("Score")            
    plt.title("Precision-Recall vs Threshold")
    plt.show()

# Function: Plot the Receiver Operating Characteristic (ROC) curve
def plot_roc_curve(fpr, tpr, label=None):
    # Plot FPR vs TPR (Recall)
    plt.plot(fpr, tpr, linewidth=2, label=label)
    # Plot the diagonal 'random guess' line
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])         # Lock axis from 0 to 1
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (Recall)')
    plt.grid(True)
    plt.title("ROC Curve")
    plt.show()

# 3. Execute plotting
plot_precision_recall_vs_threshold(precisions, recalls, thresholds_pr)
plot_roc_curve(fpr, tpr, label="Midterm Review Model")

# =============
# Exercise 4
# =============

# 1. Generate synthetic linear data (y = 4 + 3x + noise)
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# 2. Add bias term (x0 = 1) to each instance
X_b = np.c_[np.ones((100, 1)), X] 

# 3. Model Parameters
m = len(X_b)         # Number of instances
n_epochs = 50        # Number of passes over the training set
t0, t1 = 5, 50       # Learning schedule hyperparameters

def learning_schedule(t):
    """Gradually reduces the learning rate to ensure convergence."""
    return t0 / (t + t1)

# 4. Random Initialization
theta = np.random.randn(2, 1) # Weights for bias and feature

# 5. SGD Training Loop
for epoch in range(n_epochs):
    for i in range(m):
        # Pick a random instance from the dataset (Stochastic)
        random_index = np.random.randint(m)
        xi = X_b[random_index : random_index + 1]
        yi = y[random_index : random_index + 1]
        
        # Calculate gradients for the single instance
        # Gradient of MSE = 2 * X.T * (X * theta - y)
        gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
        
        # Adjust learning rate based on current iteration
        t = epoch * m + i
        eta = learning_schedule(t)
        
        # Update parameters by moving against the gradient
        theta = theta - eta * gradients

print(f"Final Theta (Intercept, Slope): \n{theta}")
        
# =============
# Exercise 5
# =============

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

# 1. Load a dataset
iris = load_iris()
X = iris.data
y = iris.target # y has multiple classes, ideal for StratifiedKFold

# 2. Define a base classifier
base_clf = LogisticRegression(solver='liblinear', random_state=42)

# 3. Initialize the StratifiedKFold
# We'll use 5 splits this time for demonstration
skfolds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("Starting manual cross-validation for Iris dataset:")

# List to store accuracy scores for each fold
accuracy_scores = []

# 4. Loop through the splits
for fold_idx, (train_index, test_index) in enumerate(skfolds.split(X, y)):
    print(f"\n--- Fold {fold_idx + 1} ---")

    # Clone the classifier for a fresh start
    clone_clf = clone(base_clf)

    # Slice the data using the indices
    X_train_fold = X[train_index]
    y_train_fold = y[train_index]
    X_test_fold = X[test_index]
    y_test_fold = y[test_index]

    # Fit the model
    clone_clf.fit(X_train_fold, y_train_fold)

    # Make predictions
    y_pred = clone_clf.predict(X_test_fold)

    # Calculate accuracy
    n_correct = sum(y_pred == y_test_fold)
    accuracy = n_correct / len(y_pred)
    accuracy_scores.append(accuracy)

    print(f"Number of correct predictions: {n_correct}/{len(y_pred)}")
    print(f"Accuracy for this fold: {accuracy:.4f}")

# 5. Print overall results
print("\n--- Summary ---")
print(f"Individual fold accuracies: {np.round(accuracy_scores, 4)}")
print(f"Mean cross-validation accuracy: {np.mean(accuracy_scores):.4f}")
print(f"Standard deviation: {np.std(accuracy_scores):.4f}")

# =============
# Exercise 6
# =============

# 1. Define the data (flattened MATLAB matrix into 25 observations)
y = np.array([5.0291, 6.5099, 5.3666, 4.1272, 4.2948,
              6.1261, 12.5140, 10.0502, 9.1614, 7.5677,
              7.2920, 10.0357, 11.0708, 13.4045, 12.8415,
              11.9666, 11.0765, 11.7774, 14.5701, 17.0440,
              17.0398, 15.9069, 15.4850, 15.5112, 17.65])
t = np.arange(1, 26) # 25 equally spaced values

# 2. Fit the data with a straight line (polyfit degree 1)
# y = m*t + c
m, c = np.polyfit(t, y, 1)
fit_line = m * t + c

# 3. Calculate residuals (Actual - Predicted)
residuals = y - fit_line

# 4. Plot the data and the fit
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(t, y, 'o', label='Original Data')
plt.plot(t, fit_line, '-', label=f'Fit: y={m:.3f}t + {c:.3f}')
plt.title('Data Fit')
plt.legend()
plt.grid(True)

# 5. Plot the residuals
plt.subplot(1, 2, 2)
plt.stem(t, residuals)
plt.axhline(0, color='black', linestyle='--')
plt.title('Residuals')
plt.grid(True)

plt.tight_layout()
plt.show()

print(f"Slope (m): {m:.4f}")
print(f"Intercept (c): {c:.4f}")

# =============
# Exercise 7
# =============

import numpy as np

def my_lin_regression(f, x, y):
    """
    Performs linear regression given a list of basis functions.
    
    Parameters:
    f : list of callable functions
        Each function should take an array and return an array (e.g., [lambda x: x**0, lambda x: x**1])
    x : array-like
        Input data points
    y : array-like
        Target values
        
    Returns:
    w : ndarray
        Calculated weights for the basis functions
    """
    x = np.array(x)
    y = np.array(y)
    n_samples = len(x)
    n_features = len(f)
    
    # Construct the design matrix (Phi)
    # Shape: (N, M) where N is number of samples, M is number of basis functions
    phi = np.zeros((n_samples, n_features))
    
    for j in range(n_features):
        phi[:, j] = f[j](x)
        
    # Solve the Normal Equation: w = (Phi^T * Phi)^-1 * Phi^T * y
    # np.linalg.lstsq is numerically more stable than explicit inversion
    w, residuals, rank, s = np.linalg.lstsq(phi, y, rcond=None)
    
    return w

# 1. Update basis functions to include a constant for the intercept
f = [np.sin, np.cos, lambda x: np.ones_like(x)]

# 2. Generate data
x = np.linspace(0, 2*np.pi, 1000)
# Adding noise with mean 0 for a cleaner regression (using normal distribution)
y = 3*np.sin(x) - 2*np.cos(x) + 0.5 + np.random.normal(0, 0.2, len(x))

# 3. Perform regression
beta = my_lin_regression(f, x, y)

# 4. Plot
plt.figure(figsize=(10, 8))
plt.plot(x, y, "b.", label="data", alpha=0.3)

# Now beta[0], beta[1], and beta[2] all exist
regression_model = beta[0]*f[0](x) + beta[1]*f[1](x) + beta[2]*f[2](x)
plt.plot(x, regression_model, "r-", label="regression", linewidth=3)

plt.xlabel("x")
plt.ylabel("y")
plt.title("Least Square Regression with Basis Functions")
plt.grid(True)
plt.legend()
plt.show()

# ===========
# Exercise 7
# ===========

def flipper(x):
    # numberOfCoins is the length of our array
    number_of_coins = len(x)
    bit_to_flip = np.random.randint(0, number_of_coins)
    x[bit_to_flip] = 1 - x[bit_to_flip]
    return x

# Simulation Parameters
N = 20  # Total coins
steps = 10000
x = np.zeros(N)  # Start with all tails (0 heads)

# Record the number of heads at each step
history = []

for _ in range(steps):
    x = flipper(x)
    history.append(np.sum(x))

# Plotting the results
plt.figure(figsize=(10, 6))
plt.hist(history, bins=range(N + 2), density=True, alpha=0.7, color='skyblue', edgecolor='black')
plt.title("Distribution of Macrostate (Number of Heads) over 10,000 Flips")
plt.xlabel("Number of Heads (n)")
plt.ylabel("Probability")
plt.grid(axis='y', alpha=0.5)
plt.show()
