import numpy as np
import pandas as pd

# Define the aridity index function
def aridity_index(pet, p):
    return pet / p

# Define the Budyko equation
def budyko(dryness_index, runoff_coefficient):
    return 1 - np.exp(-dryness_index * runoff_coefficient)

# Load precipitation, potential evapotranspiration, and observed runoff data
p = np.loadtxt("precipitation.txt")
pet = np.loadtxt("potential_evapotranspiration.txt")
runoff_obs = np.loadtxt("observed_runoff.txt")

# Calculate the aridity index for each point
aridity = aridity_index(pet, p)

# Estimate the dryness index from the aridity index
dryness = np.mean(aridity)

# Estimate the runoff coefficient from the observed runoff data
runoff_coeff = np.mean(runoff_obs) / np.mean(p)

# Calculate the long-term average annual runoff using the Budyko equation
runoff_budyko = budyko(aridity, runoff_coeff) * p

# Compare the simulated runoff with the observed runoff data
mse = np.mean((runoff_budyko - runoff_obs)**2)
print("Mean squared error: ", mse)

# DRIED UP REGION

# Load precipitation, potential evapotranspiration, and land use data
data = pd.read_csv('data.csv', index_col=0)
P = data['Precipitation'].values # Mean annual precipitation (mm/yr)
PET = data['Potential Evapotranspiration'].values # Mean annual potential evapotranspiration (mm/yr)

# CAN BE CHANGED BASED ON DRIED UP LAKES
LU = data['Land Use'].values # Land use codes (0-4)

# Assign CN values to land use types
# CN values can be found in lookup tables based on land use and soil type
CN_table = np.array([[0, 77], [1, 76], [2, 86], [3, 93], [4, 95]]) # Example CN values for land use codes 0-4
CN = np.zeros_like(LU)
for i, code in enumerate(np.unique(LU)):
    CN[LU==code] = CN_table[CN_table[:,0]==code, 1]

# Compute weighted CN value for the catchment
pct_LU = np.bincount(LU) / len(LU) # Proportion of each land use type in the catchment
CN_weighted = np.sum(CN * pct_LU) # Weighted CN value

# Estimate catchment storage capacity (S)
S = (1000 / CN_weighted) - 10 # Estimated storage capacity in mm (S=1000/CN-10)

# ESTIMATE beta USING OBSERVED VALUES
beta = 1

# Compute aridity index (AI) and runoff coefficient (R/P)
AI = PET / P
f_AI = 1 / (1 + (S/P) * (AI/1.0)**beta) # Budyko function with beta parameter
R_P = 1 - f_AI * (1 + (S/P) * (AI/1.0)**beta)

# Compute mean annual runoff and actual evapotranspiration
R = R_P * P # Mean annual runoff (mm/yr)
AET = PET - R # Mean annual actual evapotranspiration (mm/yr)