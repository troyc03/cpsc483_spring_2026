import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
import os

# Exercise 1: SVR Hyperparameter Tuning and Comparison with Linear Regression

# Function to prepare country stats (from the book)
def prepare_country_stats(oecd_bli, gdp_per_capita):
    oecd_bli = oecd_bli[oecd_bli["INEQUALITY"]=="TOT"]
    oecd_bli = oecd_bli.pivot(index="Country", columns="Indicator", values="Value")
    gdp_per_capita.rename(columns={"2015": "GDP per capita"}, inplace=True)
    gdp_per_capita.set_index("Country", inplace=True)
    full_country_stats = pd.merge(left=oecd_bli, right=gdp_per_capita,
                                  left_index=True, right_index=True)
    full_country_stats.sort_values(by="GDP per capita", inplace=True)
    remove_indices = [0, 1, 6, 8, 33, 34, 35]
    keep_indices = list(set(range(36)) - set(remove_indices))
    return full_country_stats[["GDP per capita", 'Life satisfaction']].iloc[keep_indices]

# Load the data
oecd_bli = pd.read_csv("oecd_bli_2015.csv", thousands=',')
gdp_per_capita = pd.read_csv("gdp_per_capita.csv", thousands=',', delimiter='\t',
                             encoding='latin1', na_values="n/a")

# Prepare the data
country_stats = prepare_country_stats(oecd_bli, gdp_per_capita)
X = np.c_[country_stats["GDP per capita"]]
y = np.c_[country_stats["Life satisfaction"]].ravel()  # SVR expects 1D y

print("Data prepared. Shape of X:", X.shape, "Shape of y:", y.shape)

# Try different SVR configurations
kernels = ['linear', 'rbf']
C_values = [0.1, 1, 10, 100]
gamma_values = ['scale', 'auto', 0.01, 0.1, 1]

best_score = float('inf')
best_model = None
best_params = None

results = []

for kernel in kernels:
    if kernel == 'linear':
        for C in C_values:
            svr = SVR(kernel=kernel, C=C)
            scores = cross_val_score(svr, X, y, cv=5, scoring='neg_mean_squared_error')
            mse = -scores.mean()
            results.append({'kernel': kernel, 'C': C, 'gamma': None, 'MSE': mse})
            if mse < best_score:
                best_score = mse
                best_model = svr
                best_params = {'kernel': kernel, 'C': C}
    elif kernel == 'rbf':
        for C in C_values:
            for gamma in gamma_values:
                svr = SVR(kernel=kernel, C=C, gamma=gamma)
                scores = cross_val_score(svr, X, y, cv=5, scoring='neg_mean_squared_error')
                mse = -scores.mean()
                results.append({'kernel': kernel, 'C': C, 'gamma': gamma, 'MSE': mse})
                if mse < best_score:
                    best_score = mse
                    best_model = svr
                    best_params = {'kernel': kernel, 'C': C, 'gamma': gamma}

# Print results
print("\nSVR Results:")
for result in results:
    print(f"Kernel: {result['kernel']}, C: {result['C']}, Gamma: {result['gamma']}, MSE: {result['MSE']:.4f}")

print(f"\nBest SVR Model: {best_params}")
print(f"Best Cross-Validation MSE: {best_score:.4f}")

# Train the best model on full data and make a prediction
best_model.fit(X, y)
X_new = [[22587]]  # Cyprus' GDP per capita
prediction = best_model.predict(X_new)
print(f"Prediction for Cyprus (GDP per capita: {X_new[0][0]}): {prediction[0]:.4f}")

# Compare with linear regression
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg_scores = cross_val_score(lin_reg, X, y, cv=5, scoring='neg_mean_squared_error')
lin_reg_mse = -lin_reg_scores.mean()
print(f"\nLinear Regression Cross-Validation MSE: {lin_reg_mse:.4f}")

lin_reg.fit(X, y)
lin_reg_pred = lin_reg.predict(X_new)
print(f"Linear Regression Prediction for Cyprus: {lin_reg_pred[0]:.4f}")


# Plotting the results
plt.scatter(X, y, color='red', label='Data points')
X_range = np.linspace(X.min(), X.max(), 100).reshape(-1,1)
y_svr = best_model.predict(X_range)
y_lin = lin_reg.predict(X_range)
plt.plot(X_range, y_svr, color='blue', label='SVR Prediction')
plt.plot(X_range, y_lin, color='green', label='Linear Regression Prediction')
plt.xlabel('GDP per capita')
plt.ylabel('Life satisfaction')
plt.title('SVR vs Linear Regression Predictions')
plt.legend()
plt.show()

# Exercise 2: GridSearchCV 
from sklearn.model_selection import GridSearchCV

param_grid = {
    'kernel': ['linear', 'rbf'],
    'C': [0.1,1,10,100],
    'gamma': ['scale', 'auto', 0.01,0.1,1]
}

svr = SVR()
grid_search = GridSearchCV(svr, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X,y)
print(f"\nBest parameters from GridSearchCV: {grid_search.best_params_}")
print(f"Best cross-validation MSE from GridSearchCV: {-grid_search.best_score_:.4f}")

