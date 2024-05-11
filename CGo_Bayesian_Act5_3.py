# -*- coding: utf-8 -*-
"""
Created on Tue May  7 14:30:34 2024

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

# Prior distribution (uniform distribution)
uniform_dist = sts.uniform.pdf(mu) + 1
uniform_dist = uniform_dist / uniform_dist.sum()

# Calculate unnormalized posterior probability
unnormalized_posterior = likelihood_out * uniform_dist

# Plot unnormalized posterior
plt.plot(mu, unnormalized_posterior)
plt.xlabel("$\mu$ in meters")
plt.ylabel("Unnormalized Posterior")
plt.grid(True)
plt.show()


