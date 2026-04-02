import numpy as np
import time
from src.algorithms.base_optimizer import BaseOptimizer

class ABC(BaseOptimizer):
    """
    Artificial Bee Colony (ABC) implementation.
    Focuses on localized exploitation (Employed/Onlooker) and global exploration (Scout).
    Inherits administrative functions from BaseOptimizer. [cite: 101, 245]
    """
    def __init__(self, objective_func, bounds, dim, pop_size, max_fes, seed, limit=None):
        super().__init__(objective_func, bounds, dim, pop_size, max_fes, seed)
        
        # In ABC, food sources are pop_size / 2
        self.food_number = self.pop_size // 2
        # Limit for abandoning a food source (Standard: FoodNumber * Dimension) [cite: 110]
        self.limit = limit if limit is not None else (self.food_number * self.dim)
        
        # Trial counters for each food source
        self.trials = np.zeros(self.food_number)

    def optimize(self):
        start_time = time.time()
        
        # 1. Initialization: Food sources represent potential solutions 
        # We adjust pop_size to food_number for internal operations
        lower_bound, upper_bound = self.bounds
        foods = np.random.uniform(lower_bound, upper_bound, (self.food_number, self.dim))
        fitness = self.evaluate_fitness(foods) # BaseOptimizer updates best_fitness here
        
        while self.fes_counter < self.max_fes:
            
            # --- EMPLOYED BEES PHASE --- 
            for i in range(self.food_number):
                # Select a random neighbor (k != i)
                k = np.random.choice([idx for idx in range(self.food_number) if idx != i])
                phi = np.random.uniform(-1, 1, self.dim)
                
                # Generate candidate food source [cite: 113]
                candidate = foods[i] + phi * (foods[i] - foods[k])
                candidate = self.enforce_boundaries(np.atleast_2d(candidate))
                
                candidate_fit = self.evaluate_fitness(candidate)[0]
                
                # Greedy Selection 
                if candidate_fit < fitness[i]:
                    foods[i] = candidate[0]
                    fitness[i] = candidate_fit
                    self.trials[i] = 0
                else:
                    self.trials[i] += 1

            # --- ONLOOKER BEES PHASE --- 
            # Probabilities based on fitness (Roulette Wheel) 
            # Note: For minimization, we use inverse or offset fitness
            fit_weights = 1.0 / (1.0 + fitness)
            prob = fit_weights / np.sum(fit_weights)
            
            m = 0
            n = 0
            while m < self.food_number:
                if np.random.rand() < prob[n]:
                    # Same search logic as employed bees 
                    k = np.random.choice([idx for idx in range(self.food_number) if idx != n])
                    phi = np.random.uniform(-1, 1, self.dim)
                    
                    candidate = foods[n] + phi * (foods[n] - foods[k])
                    candidate = self.enforce_boundaries(np.atleast_2d(candidate))
                    candidate_fit = self.evaluate_fitness(candidate)[0]
                    
                    if candidate_fit < fitness[n]:
                        foods[n] = candidate[0]
                        fitness[n] = candidate_fit
                        self.trials[n] = 0
                    else:
                        self.trials[n] += 1
                    m += 1
                n = (n + 1) % self.food_number

            # --- SCOUT BEES PHASE --- 
            # Abandon source if limit is exceeded 
            abandoned_idx = np.where(self.trials >= self.limit)[0]
            for idx in abandoned_idx:
                # Replace with a completely new random source (Global Exploration) 
                foods[idx] = np.random.uniform(lower_bound, upper_bound, self.dim)
                fitness[idx] = self.evaluate_fitness(np.atleast_2d(foods[idx]))[0]
                self.trials[idx] = 0

        execution_time = time.time() - start_time
        is_successful = 1 if self.best_fitness <= 1e-8 else 0
        
        return {
            "Algorithm_Name": "ABC",
            "Function_Name": self.objective_func.__name__,
            "Dimension": self.dim,
            "Run_ID": None,
            "Random_Seed": self.seed,
            "Hyperparameters": {"limit": self.limit, "food_number": self.food_number},
            "Convergence_Data": {
                "FEs_Milestones": self.fes_milestones,
                "Fitness_Trajectory": self.convergence_curve
            },
            "Final_Evaluation": {
                "Final_Best_Fitness": self.best_fitness,
                "Final_Best_Position": self.best_position.tolist(),
                "Is_Successful": is_successful,
                "Execution_Time_Sec": execution_time
            }
        }