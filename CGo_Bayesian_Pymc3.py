# -*- coding: utf-8 -*-
"""
Created on Tue May  7 14:32:17 2024

@author: CGo
"""
import pymc3 as pm
import numpy as np
import matplotlib.pyplot as plt

# Generate some synthetic data
np.random.seed(0)
X = np.linspace(0, 10, 100)
true_slope = 2
true_intercept = 1
y = true_slope * X + true_intercept + np.random.normal(0, 1, size=len(X))

# Define the probabilistic model
with pm.Model() as linear_model:
    # Priors for regression coefficients
    slope = pm.Normal("slope", mu=0, sigma=10)
    intercept = pm.Normal("intercept", mu=0, sigma=10)
    
    # Likelihood (sampling distribution) of observations
    likelihood = pm.Normal("y", mu=slope*X + intercept, sigma=1, observed=y)
    
    # Perform Bayesian inference
    trace = pm.sample(1000, tune=1000)

# Plot the results
plt.figure(figsize=(10, 6))
plt.scatter(X, y, label="Data")
plt.xlabel("X")
plt.ylabel("y")
plt.title("Bayesian Linear Regression")
pm.plot_posterior_predictive_glm(trace, samples=100, eval=np.linspace(0, 10, 100), label="Posterior Predictive Regression Lines")
plt.plot(X, true_slope * X + true_intercept, label="True Regression Line", color="red", linestyle="--")
plt.legend()
plt.show()
