# -*- coding: utf-8 -*-
"""
Created on Tue May  7 14:28:05 2024

@author: CGo
"""
import scipy.stats as sts
import numpy as np
import matplotlib.pyplot as plt

# Observation value
datum = 1.7

# Define x-axis values (mu values)
mu = np.linspace(1.65, 1.8, num=50)

# Calculate likelihood function
likelihood_out = sts.norm.pdf(datum, mu, scale=0.1)
likelihood_out = likelihood_out / likelihood_out.sum()

# Generate plot
plt.plot(mu, likelihood_out)
plt.title("Likelihood of $\mu$ given observation 1.7m")
plt.ylabel("Probability Density/Likelihood")
plt.xlabel("Value of $\mu$")
plt.grid(True)
plt.show()

