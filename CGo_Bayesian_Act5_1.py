# -*- coding: utf-8 -*-
"""
Created on Tue May  7 14:26:17 2024

@author: CGo
"""
import scipy.stats as sts
import numpy as np
import matplotlib.pyplot as plt

# Define the x-axis values
mu = np.linspace(1.65, 1.8, num=50)

# Create the uniform distribution
uniform_dist = sts.uniform.pdf(mu) + 1
uniform_dist = uniform_dist / uniform_dist.sum()

# Create the beta distribution
beta_dist = sts.beta.pdf(mu, 2, 5, loc=1.65, scale=0.2)
beta_dist = beta_dist / beta_dist.sum()

# Plot the distributions
plt.plot(mu, beta_dist, label='Beta Dist')
plt.plot(mu, uniform_dist, label='Uniform Dist')

# Add labels and title
plt.xlabel("Value of $\mu$ in meters")
plt.ylabel("Probability density")
plt.title("Probability Density Functions")

# Add legend
plt.legend()

# Show the plot
plt.grid(True)
plt.show()


