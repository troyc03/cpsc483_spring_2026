# -*- coding: utf-8 -*-
"""
File name: cpsc483_midterm_review_3.py
Purpose: This file contains all computational work
for the second practice exam for CPSC 483 (Intro to
Machine Learning) and revisited solutions of HW1 and HW2.
"""

# =============
# Exercise 1
# =============

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Given values
p_d = 0.0005 # P(D)
p_pos_given_d = 0.97 # P(+|D)
p_neg_given_not_d = 0.95 # P(-|not D)
p_pos_given_not_d = 1 - 0.97 # P(+|not D)
p_neg_given_d = 1 - 0.95 # P(-|D)

# Population size
population_size = 10000 
actual = np.zeros(population_size) # Actual numerical values
predicted = np.zeros(population_size) # Predicted numerical values

# Populate actual diseases
num_diseased = int(p_d * population_size)
actual[num_diseased:] = 0 # Extract values with no disease
actual[:num_diseased] = 1 # Extract values with disease

# Populate predictions based on probabilities
# Diseased group
actual_d = actual == 1 # Actual disease is True
actual_not_d = actual == 0 # Actual complement is False

# Ensure this sums to 1
probs = [p_pos_given_not_d, 1 - p_pos_given_not_d] # Probabilities
predicted[actual_not_d] = np.random.choice([1, 0], size=sum(actual_not_d), p=probs) \
# Normalized probs

# 2. Compute confusion matrix
cm = confusion_matrix(actual, predicted)

# 3. Visualize confusion matrix
plt.figure(figsize=(8, 6)) 
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', 
            xticklabels=['Negative', 'Positive'], 
            yticklabels=['No Disease', 'Disease'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix: Diagnostic Test')
plt.show()

"""
Note that the "hotter" the matrix gets the
higher the count of observations in that cell.
We have a high count of people who tested positive (TP)
for no disease.
"""

# Now we can compute the probabilities
p_d = 0.0005
p_not_d = 1 - p_d

p_pos_given_d = 0.97
p_neg_given_not_d = 0.95

# Corrected complements
p_pos_given_not_d = 1 - p_neg_given_not_d 
p_neg_given_d = 1 - p_pos_given_d         

# Total probabilities (Normalization)
p_pos = (p_pos_given_d * p_d) + (p_pos_given_not_d * p_not_d)
p_neg = (p_neg_given_d * p_d) + (p_neg_given_not_d * p_not_d)

# Posterior probabilities
p_d_given_pos = (p_pos_given_d * p_d) / p_pos
p_d_given_neg = (p_neg_given_d * p_d) / p_neg

print(f"Probability: {p_d_given_pos}"); print(f"Probability: {p_d_given_neg}")

# =============
# Exercise 2
# =============

import math

k = 3 # Number of flips
n = 5 # Number of total outcomes

def likelihood(theta):
    if theta < 0 or theta > 1:
        return 0.0
    else:
        return math.comb(n, k) * (theta**k) * (1 - theta)**(n - k)

# Log likelihood (after deriving)
theta_mle = k / n

# Grid check
thetas = np.linspace(0, 1, 1001) # This creates a grid space between 0, 1
Ls = np.array([likelihood(t) for t in thetas]) # Likelihood values 
theta_grid = thetas[np.argmax(Ls)]
print('-' * 50)
print(f'Maximum Likelihood: {theta_mle}')
print(f'Grid of values: {theta_grid}')
print(f'Maximum Likelihoods: {Ls.max()}')
print('-' * 50)

# =============
# Exercise 3
# =============

# Q3.1 Equation for log-likelihood (up to a constant)
# log_likelihood = - (1 / (2 * sigma^2)) * sum((x_i - mu)^2)

# Q3.2 Equation for mu_estimate (MLE)
# mu_hat = np.mean(X)

# Given example dataset
X = np.array([2.1, 1.9, 2.4, 2.0, 1.8, 2.2, 2.3, 1.7], dtype=float)
# Known variance
sigma2 = 0.25 # (sigma = 0.5) # Population standard deviation

# TODO: compute mu_hat (MLE)
mu_hat = np.mean(X) # Population mean
print(f"Calculated mu_hat: {mu_hat}")

# Visualization setup
mus = np.linspace(X.min()-1.0, X.max()+1.0, 400)
# logL calculation based on Q3.1, removing constant terms for optimization
logL = - (1/(2*sigma2)) * np.array([np.sum((X - m)**2) for m in mus])

# Visualization
plt.figure()
plt.plot(mus, logL) # Plot normalized values and log likelihood
plt.axvline(mu_hat, linestyle='--', color='red', label=f'MLE: {mu_hat}') 
# Population mean is at the peak of the bell curve
plt.xlabel("mu")
plt.ylabel("log-likelihood (up to a constant)")
plt.title("Maximum Likelihood for a 1D Gaussian")
plt.legend()
plt.grid(True)
plt.show()

print(f"MLE Verification: {mus[np.argmax(logL)]}, {mu_hat}")

print('-' * 50)

# =============
# Exercise 4
# =============

import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc

# 1. Load the dataset
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, random_state=0
)

# 2. Train Models
# 'probability=True' is required for SVC to use predict_proba for the ROC curve
svm = SVC(probability=True, random_state=0).fit(X_train, y_train)
log_reg = LogisticRegression(max_iter=10000, random_state=0).fit(X_train, y_train)

# 3. Predictions and Basic Evaluation
y_pred_svm = svm.predict(X_test)
print(f"SVM Accuracy (Test): {accuracy_score(y_test, y_pred_svm):.3f}")
print("\nClassification Report (SVM):\n", classification_report(y_test, y_pred_svm, target_names=cancer.target_names))

# 4. Confusion Matrix (SVM)
# ConfusionMatrixDisplay is a convenient way to visualize the confusion matrix
cm = confusion_matrix(y_test, y_pred_svm)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=cancer.target_names)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix (SVM)")
plt.show()

# 5. ROC Curve Computation
# We use the probability of the positive class (column 1)
y_prob_svm = svm.predict_proba(X_test)[:, 1]
y_prob_log = log_reg.predict_proba(X_test)[:, 1]

fpr_svm, tpr_svm, _ = roc_curve(y_test, y_prob_svm)
fpr_log, tpr_log, _ = roc_curve(y_test, y_prob_log)

# 6. Plotting ROC Curves
plt.figure(figsize=(8, 6))
plt.plot(fpr_svm, tpr_svm, label=f'SVM (AUC = {auc(fpr_svm, tpr_svm):.2f})')
plt.plot(fpr_log, tpr_log, label=f'Logistic Regression (AUC = {auc(fpr_log, tpr_log):.2f})')
plt.plot([0, 1], [0, 1], 'k--') # Diagonal random-guess line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.show()

# =============
# Exercise 5
# =============

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# 1. Generate synthetic correlated data
rng = np.random.RandomState(1)
X = np.dot(rng.rand(2, 2), rng.randn(2, 200)).T
X[:, 0] -= 2  # Shift to match the "Original data" centering in the image

# 2. Fit PCA
pca = PCA(n_components=2)
pca.fit(X)
X_pca = pca.transform(X)

# 3. Component dropping (dimensionality reduction)
# We set the second component to zero
X_pca_dropped = X_pca.copy()
X_pca_dropped[:, 1] = 0

# 4. Back-rotation (inverse transform)
X_new = pca.inverse_transform(X_pca_dropped)

# --- Plotting ---
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
plt.subplots_adjust(hspace=0.3, wspace=0.3)

# Top Left: Original Data
axes[0, 0].scatter(X[:, 0], X[:, 1], c=X[:, 0], cmap='viridis')
axes[0, 0].set_title("Original data")
axes[0, 0].set_xlabel("feature 1")
axes[0, 0].set_ylabel("feature 2")
# Draw eigenvectors
for length, vector in zip(pca.explained_variance_, pca.components_):
    v = vector * 3 * np.sqrt(length)
    axes[0, 0].annotate('', pca.mean_ + v, pca.mean_, 
                        arrowprops=dict(arrowstyle='->', linewidth=2, color='black'))

# Top Right: Transformed Data
axes[0, 1].scatter(X_pca[:, 0], X_pca[:, 1], c=X[:, 0], cmap='viridis')
axes[0, 1].set_title("Transformed data")
axes[0, 1].set_xlabel("First principal component")
axes[0, 1].set_ylabel("Second principal component")

# Bottom Left: Transformed data w/ second component dropped
axes[1, 0].scatter(X_pca_dropped[:, 0], X_pca_dropped[:, 1], c=X[:, 0], cmap='viridis')
axes[1, 0].set_title("Transformed data w/ second component dropped")
axes[1, 0].set_xlabel("First principal component")

# Bottom Right: Back-rotation
axes[1, 1].scatter(X_new[:, 0], X_new[:, 1], c=X[:, 0], cmap='viridis')
axes[1, 1].set_title("Back-rotation using only first component")
axes[1, 1].set_xlabel("feature 1")
axes[1, 1].set_ylabel("feature 2")

for ax in axes.flat:
    ax.set_xlim(-8, 8)
    ax.set_ylim(-8, 8)

plt.show()

"""
This is a PCA model. The most significant use of PCA is reducing the number of
variables in a dataset while keeping as much "information" (variance) as possible.
"""


