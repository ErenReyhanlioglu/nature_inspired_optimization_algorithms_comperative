import numpy as np

DIMENSIONS = [10, 20, 30]  # Scalability test levels 
RUNS_PER_SCENARIO = 30     # Independent runs for statistical significance 

#  pop_size = D * POP_SIZE_MULTIPLIER
POP_SIZE_MULTIPLIER = 5

SUCCESS_THRESHOLD = 1e-8

# FEs = 10,000 * D
MAX_FES_PER_DIM = {
    10: 100000,
    20: 200000,
    30: 300000
}

# 30 unique seeds to be used across all 18 scenarios
RANDOM_SEEDS = [
    42, 101, 202, 303, 404, 505, 606, 707, 808, 909,
    111, 222, 333, 444, 555, 666, 777, 888, 999, 1010,
    123, 456, 789, 987, 654, 321, 543, 210, 135, 246
]

HYPERPARAMETERS = {
    "PSO": {"w": 0.7298, "c1": 1.49618, "c2": 1.49618},
    "DE": {"F": 0.5, "CR": 0.9}, 
    "GWO": {"a_start": 2.0, "a_end": 0.0}, 
    "ABC": {"limit": None}, 
    # lambd is set to None to dynamically match the population size at runtime. 
    # mu will be calculated as lambd // 7 (Schwefel 1/7 rule).
    "ES": {"mu": None, "lambd": None, "sigma_init": None} 
}

FUNCTION_CONFIG = {
    "sphere": {"bounds": [-100.0, 100.0], "optimum": 0.0},
    "ackley": {"bounds": [-32.768, 32.768], "optimum": 0.0},
    "zakharov": {"bounds": [-5.0, 10.0], "optimum": 0.0}
}