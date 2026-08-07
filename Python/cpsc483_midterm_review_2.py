# -*- coding: utf-8 -*-
"""
File name: cpsc483_midterm_review_2.py
Purpose: This file contains all computational work
for the second practice exam for CPSC 483 (Intro to
Machine Learning). 
"""

# =============
# Exercise 1
# =============

import numpy as np
import matplotlib.pyplot as plt

# Reproducibility
np.random.seed(1)

# Generate data (same shape as figure)
x = np.linspace(-3, 3, 80)
y = 0.8*x**2 + 0.5*x + 2 + np.random.randn(len(x)) * 0.8

# Degrees to match figure
degrees = [1, 2, 30]  # 30 gives those vertical spikes

colors = ['red', 'blue', 'green']
linestyles = ['-', '--', '-']

plt.figure(figsize=(6,4))

# Scatter data (blue dots)
plt.scatter(x, y, color='blue', s=10)

# Fit and plot each polynomial
x_fit = np.linspace(-3, 3, 500)

for d, c, ls in zip(degrees, colors, linestyles):
    # Vandermonde matrix
    A = np.vander(x, N=d+1, increasing=True)
    
    # Solve (use pseudo-inverse for stability at high degree)
    w = np.linalg.pinv(A) @ y
    
    # Prediction
    A_fit = np.vander(x_fit, N=d+1, increasing=True)
    y_fit = A_fit @ w
    
    plt.plot(x_fit, y_fit, color=c, linestyle=ls, label=f"{d}")

# Axes + legend to match style
plt.xlabel(r"$x_1$")
plt.ylabel("y")
plt.ylim(0, 10)
plt.legend(title=None)
plt.grid(True)
plt.title("High-degree Polynomial Regression")

plt.show()

# =============
# Exercise 2
# =============

from sklearn.datasets import load_digits
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

# 1. Load small dataset (FAST)
digits = load_digits()
X, y = digits.data, digits.target

# 2. Scale features
scaler = StandardScaler() # Recall that the StandardScaler standardizes \
    # numerical features so that they have a mean of zero and a
    # standard deviation of one
X_scaled = scaler.fit_transform(X.astype(np.float64)) # Convert \
    # numerical features to float values

# 3. Train a Stochastic Gradient Descent model
sgd_clf = SGDClassifier(random_state=42)

# Cross-validation accuracy
scores = cross_val_score(sgd_clf, X_scaled, y, cv=3, scoring="accuracy")
print("Accuracy:", scores) # Output accuracy values

# 4. Predictions via CV (This will calculate TP, FP, TN, FN)
y_pred = cross_val_predict(sgd_clf, X_scaled, y, cv=3)

# 5. Build the confusion matrix 
conf_mx = confusion_matrix(y, y_pred)
print(conf_mx)

# 6. Plot results
plt.matshow(conf_mx, cmap=plt.cm.gray)
plt.colorbar()
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# =============
# Exercise 3
# =============

import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.metrics import classification_report

# Let's use the Iris dataset from before
iris = datasets.load_iris()
print(iris.feature_names) #Output feature names
print(iris.data[:10]) # Output the first ten values
print('We have %d data samples with %d \
    features'%(iris.data.shape[0], iris.data.shape[1])) 
    # This generates a data matrix 

print(iris.target_names) # Output target names
print(set(iris.target)) # Set up a target column vector

# Let's visualize two features for now.
X = iris.data[:, [0, 2]]
y = iris.target
target_names = iris.target_names
colors = ['b', 'g', 'r']
symbols = ['o', '^', '*']

# Get the classes from the data matrix
n_class = len(set(y))
print('We have %d classes in the data'%(n_class))
plt.figure(figsize = (10,8))

# Let's have a look of the data first
for i, c, s in (zip(range(n_class), colors, symbols)):
    ix = y == i
    plt.scatter(X[:, 0][ix], X[:, 1][ix], \
                color = c, marker = s, s = 60, \
                label = target_names[i])
    plt.title("Support Vector Machine Model")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")


# Initialize SVM classifier
clf = svm.SVC(kernel='linear')

# Train the classifier with data
clf.fit(X,y)

# Predict on the data
clf.predict(X)

def plot_decision_boundary(X, y, clf, title = None):
    '''
    Helper function to plot the decision boundary for the SVM
    '''
    
    # Define features and the target values of the SVM
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    
    # Predict the support vectors for the model
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot results
    plt.figure(figsize = (10,8))
    plt.contour(xx, yy, Z, alpha=0.4)
    
    for i, c, s, in zip(range(n_class), colors, symbols):
        ix = y == i
        plt.scatter(X[:, 0][ix], X[:, 1][ix], color = c, marker = s, s = 60, 
                    label=target_names[i])
    
    if title is not None:
        plt.title(title)
    
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title("Support Vector Machine Model (With Decision Boundaries")
    plt.legend()
    plt.show()
    
plot_decision_boundary(X, y, clf) 

"""
The support vectors are the 
points closest to the hyperplane/decision 
boundary.

"""

# =============
# Exercise 4
# =============

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer

# Load dataset
cancer = load_breast_cancer()

# Split dataset by class
malignant = cancer.data[cancer.target == 0]
benign = cancer.data[cancer.target == 1]

# Create subplots
malignant = cancer.data[cancer.target == 0]
benign = cancer.data[cancer.target == 1]

# Create subplots (15 rows, 2 columns = 30 features)
fig, axes = plt.subplots(15, 2, figsize=(10, 20))
ax = axes.ravel()

# Plot histograms
for i in range(30):
    _, bins = np.histogram(cancer.data[:,i], bins = 50)
    ax[i].hist(malignant[:, i], bins=bins, color='blue', alpha=0.5)
    ax[i].hist(benign[:, i], bins=bins, color='orange', alpha=.5)
    ax[i].set_title(cancer.feature_names[i])
    ax[i].set_yticks(())
ax[0].set_xlabel("Features magnitude")
ax[0].set_ylabel("Frequency")
ax[0].legend(["malignant", "benign"], loc="best")
fig.tight_layout()

# Show histograms
plt.show()

"""
We are performing exploratory data analysis in this example,
however this can be used for classification models such as
logistic regression.
"""