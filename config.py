import numpy as np

# 1. Global Simulation Constants
DIMENSIONS = [10, 30, 50]  # Scalability test levels 
RUNS_PER_SCENARIO = 30     # Independent runs for statistical significance 
POP_SIZE = 30              # Fixed population size for all algorithms 

# Maximum Function Evaluations (Max FEs) based on dimension 
# Standard: D * 10,000 
MAX_FES_PER_DIM = {
    10: 100000,
    30: 300000,
    50: 500000
}

# 2. Random Seeds (Deterministic for reproducibility) 
# 30 unique seeds to be used across all 18 scenarios
RANDOM_SEEDS = [
    42, 101, 202, 303, 404, 505, 606, 707, 808, 909,
    111, 222, 333, 444, 555, 666, 777, 888, 999, 1010,
    123, 456, 789, 987, 654, 321, 543, 210, 135, 246
]

# 3. Algorithm Hyperparameters 
# These values are based on standard literature recommendations
HYPERPARAMETERS = {
    "PSO": {"w": 0.7298, "c1": 1.49618, "c2": 1.49618}, # Standard Clerc-Kennedy settings
    "DE": {"F": 0.5, "CR": 0.9}, # Common DE/rand/1/bin defaults 
    "GWO": {"a_start": 2.0, "a_end": 0.0}, # Linear decay of a 
    "ABC": {"limit": None}, # If None, will be calculated as (FoodNumber * D) 
    "RCGA": {"pc": 0.8, "pm": 0.1, "tournament_size": 3}, # 
    "ES": {"mu": 7, "lambd": 30, "sigma_init": None} # (mu + lambda) strategy 
}

# 4. Function Meta-Data (Bounds and Global Optima) 
FUNCTION_CONFIG = {
    "sphere": {"bounds": [-100.0, 100.0], "optimum": 0.0},
    "rastrigin": {"bounds": [-5.12, 5.12], "optimum": 0.0},
    "rosenbrock": {"bounds": [-30.0, 30.0], "optimum": 0.0}
}