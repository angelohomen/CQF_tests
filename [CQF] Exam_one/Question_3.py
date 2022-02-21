# CQF Program - Exam One
# Angelo Rafael Lenarte Homen

# QUESTION 3: CALCULATING GRADIENT VAR AND ES RELATED TO THE PORTFOLIO WEIGHTS VARIATION

# Import libraries
import numpy as np
from scipy.stats import norm

# Assets weights
w_1 = 50 / 100
w_2 = 20 / 100
w_3 = 30 / 100

# Assets expected returns
mu_1 = 0
mu_2 = 0
mu_3 = 0

# Assets volatilities
sig_1 = 30 / 100
sig_2 = 20 / 100
sig_3 = 15 / 100

# Correlations between the assets
rho_1_1 = 1
rho_1_2 = 0.8
rho_1_3 = 0.5
rho_2_2 = 1
rho_2_3 = 0.3
rho_3_3 = 1

# Volatilities diagonal matrix
diag_sig = np.array([[sig_1, 0, 0], [0, sig_2, 0], [0, 0, sig_3]])

# Correlation matrix
rho = np.array([[rho_1_1, rho_1_2, rho_1_3], [rho_1_2, rho_2_2, rho_2_3], [rho_1_3, rho_2_3, rho_3_3]])

# Matrices
w = np.array([[w_1], [w_2], [w_3]])
mu = np.array([[mu_1], [mu_2], [mu_3]])
vol = np.array([[sig_1], [sig_2], [sig_3]])

# Calculating sigma
sigma = np.dot(np.dot(diag_sig, rho), diag_sig)

# Factor
alpha = 0.99
factor = norm.ppf(1 - alpha)

# Calculating VAR
VAR = np.dot(np.transpose(w), mu) + np.dot(np.dot(factor, sigma), w) / np.sqrt(np.dot(np.dot(np.transpose(w), sigma), w))

# Calculating ES
ES = np.dot(np.transpose(w), mu) - np.dot(np.dot(norm.pdf(factor), sigma), w) / (np.dot((1 - alpha), np.sqrt(np.dot(np.dot(np.transpose(w), sigma), w))))

# Print results
for row in range(len(VAR)):
    for col in range(len(VAR[row])):
        print(f'dVaR(w)/dw{row + 1} = ', end='\t')
        print(np.round(VAR[row][col] * 100, 2), end =' %\n')
        print(f'dES(w)/dw{row + 1} = ', end='\t')
        print(np.round(ES[row][col] * 100, 2), end =' %\n\n')