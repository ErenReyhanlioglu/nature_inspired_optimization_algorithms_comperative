import numpy as np
import time
from src.algorithms.base_optimizer import BaseOptimizer

class ABC(BaseOptimizer):
    """
    Artificial Bee Colony (ABC) implementation.
    Focuses on localized exploitation (Employed/Onlooker) and global exploration (Scout).
    Corrected to update only a single random dimension per perturbation.
    """
    def __init__(self, objective_func, bounds, dim, pop_size, max_fes, seed, **kwargs):
        super().__init__(objective_func, bounds, dim, pop_size, max_fes, seed, **kwargs)

        # Standard formulation: food number is half of the total population
        self.food_number = self.pop_size // 2
        
        # Standard heuristic for the limit parameter if not provided
        if getattr(self, 'limit', None) is None:
            self.limit = self.food_number * self.dim
            
        self.hparams['limit'] = self.limit
        self.hparams['food_number'] = self.food_number
        
        self.trials = np.zeros(self.food_number)

    def optimize(self):
        """
        Executes the Artificial Bee Colony optimization loop.
        Includes Employment, Onlooker, and Scout phases with strict budget controls.
        """
        start_time = time.time()
        
        lower_bound, upper_bound = self.bounds
        foods = np.random.uniform(lower_bound, upper_bound, (self.food_number, self.dim))
        fitness = self.evaluate_fitness(foods) 
        
        while self.fes_counter < self.max_fes and self.best_fitness > self.success_threshold:
            
            # --- EMPLOYED BEES PHASE ---
            for i in range(self.food_number):
                if self.fes_counter >= self.max_fes:
                    break
                
                # Select a random partner different from the current bee
                k = np.random.choice([idx for idx in range(self.food_number) if idx != i])
                
                j = np.random.randint(0, self.dim)
                
                phi = np.random.uniform(-1, 1)

                candidate = foods[i].copy()
                candidate[j] = foods[i, j] + phi * (foods[i, j] - foods[k, j])
                
                candidate = self.enforce_boundaries(np.atleast_2d(candidate))
                candidate_fit = self.evaluate_fitness(candidate)[0]

                # Greedy selection
                if candidate_fit < fitness[i]:
                    foods[i] = candidate[0]
                    fitness[i] = candidate_fit
                    self.trials[i] = 0
                else:
                    self.trials[i] += 1

            if self.fes_counter >= self.max_fes:
                continue

            # --- ONLOOKER BEES PHASE ---
            fit_weights = 1.0 / (1.0 + fitness)
            prob = fit_weights / np.sum(fit_weights)

            m = 0
            n = 0
            while m < self.food_number:
                if self.fes_counter >= self.max_fes:
                    break
                
                # Roulette wheel selection
                if np.random.rand() < prob[n]:
                    k = np.random.choice([idx for idx in range(self.food_number) if idx != n])
                    
                    # Select a SINGLE random dimension to update (CRITICAL FIX)
                    j = np.random.randint(0, self.dim)
                    phi = np.random.uniform(-1, 1)

                    candidate = foods[n].copy()
                    candidate[j] = foods[n, j] + phi * (foods[n, j] - foods[k, j])
                    
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

            if self.fes_counter >= self.max_fes:
                continue

            # --- SCOUT BEES PHASE ---
            abandoned_idx = np.where(self.trials >= self.limit)[0]
            for idx in abandoned_idx:
                if self.fes_counter >= self.max_fes:
                    break
                
                # Replace the abandoned food source completely with a new random one
                foods[idx] = np.random.uniform(lower_bound, upper_bound, self.dim)
                fitness[idx] = self.evaluate_fitness(np.atleast_2d(foods[idx]))[0]
                self.trials[idx] = 0

        execution_time = time.time() - start_time
        
        is_successful = 1 if self.best_fitness <= self.success_threshold else 0
        
        return {
            "Algorithm_Name": "ABC",
            "Function_Name": self.objective_func.__name__,
            "Dimension": self.dim,
            "Run_ID": None,
            "Random_Seed": self.seed,
            "Hyperparameters": self.hparams,
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