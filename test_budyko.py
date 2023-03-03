import numpy as np
import matplotlib.pyplot as plt

# Define parameters
beta = 2.0
P = 1000.0
S = 200.0
PET = np.linspace(0, 2000, 100)

# Define a function to compute the Budyko curve with gamma
def budyko_gamma(AI, gamma):
    f_AI = 1 / (1 + (S/P) * (AI/1.0)**beta/gamma)
    R_P = 1 - f_AI
    return R_P

# Estimate the value of gamma using optimization
from scipy.optimize import minimize

def obj_func(gamma, AI, obs_R_P):
    pred_R_P = budyko_gamma(AI, gamma)
    return np.sum((pred_R_P - obs_R_P)**2)

obs_R_P = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
AI = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
result = minimize(obj_func, x0=0.5, args=(AI, obs_R_P))

# Plot the Budyko curve with gamma
gamma = result.x[0]
R_P = budyko_gamma(PET/P, gamma)
plt.plot(PET, R_P)
plt.xlabel('PET/P')
plt.ylabel('R/P')
plt.title('Budyko curve with gamma={:.2f}'.format(gamma))
plt.show()
