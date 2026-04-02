import numpy as np
import time
from src.algorithms.base_optimizer import BaseOptimizer

class ES(BaseOptimizer):
    """
    (mu + lambda) Evolution Strategies (ES) implementation.
    Uses Gaussian Mutation with the 1/5 Success Rule for dynamic step-size adaptation.
    Inherits administrative functions from BaseOptimizer.
    """
    def __init__(self, objective_func, bounds, dim, pop_size, max_fes, seed, 
                 mu=None, lambd=None, sigma_init=None):
        super().__init__(objective_func, bounds, dim, pop_size, max_fes, seed)
        
        # ES notation: mu = number of parents, lambd = number of offspring
        self.mu = mu if mu is not None else self.pop_size // 4
        self.lambd = lambd if lambd is not None else self.pop_size
        
        # Initial mutation step size (sigma)
        # Default: 10% of the search space range
        range_width = self.bounds[1] - self.bounds[0]
        self.sigma = sigma_init if sigma_init is not None else range_width * 0.1
        
        # Parameters for the 1/5 Success Rule
        self.k = 0.85  # Decrease factor
        self.update_interval = 10 # Check success every 'n' generations
        self.success_count = 0
        self.total_mutations = 0

    def optimize(self):
        start_time = time.time()
        
        # 1. Initialization: Create initial parents
        parents = self.initialize_population()[:self.mu]
        parent_fitness = self.evaluate_fitness(parents)
        
        generation_count = 0
        
        while self.fes_counter < self.max_fes:
            generation_count += 1
            offspring_population = []
            
            # 2. Mutation Phase: Generate 'lambd' offspring from 'mu' parents
            for _ in range(self.lambd):
                # Randomly select a parent
                parent_idx = np.random.randint(0, self.mu)
                parent = parents[parent_idx]
                
                # Apply Gaussian Mutation: x' = x + N(0, sigma)
                noise = np.random.normal(0, self.sigma, self.dim)
                offspring = parent + noise
                offspring = self.enforce_boundaries(np.atleast_2d(offspring))[0]
                
                offspring_population.append(offspring)
            
            # 3. Evaluation
            offspring_population = np.array(offspring_population)
            offspring_fitness = self.evaluate_fitness(offspring_population)
            
            # 4. Success Rule Tracking
            # Count how many offspring outperformed their random parent
            for i in range(self.lambd):
                self.total_mutations += 1
                # Check against the best parent fitness for a strict success criterion
                if offspring_fitness[i] < np.min(parent_fitness):
                    self.success_count += 1
            
            # 5. Selection (mu + lambda strategy)
            # Combine parents and offspring, then select the best 'mu'
            combined_pop = np.vstack((parents, offspring_population))
            combined_fit = np.concatenate((parent_fitness, offspring_fitness))
            
            indices = np.argsort(combined_fit)
            parents = combined_pop[indices[:self.mu]]
            parent_fitness = combined_fit[indices[:self.mu]]
            
            # 6. Self-Adaptation: 1/5 Success Rule Update
            if generation_count % self.update_interval == 0:
                ps = self.success_count / self.total_mutations
                if ps > 0.2:
                    # High success rate: Increase sigma to explore further
                    self.sigma = self.sigma / self.k
                elif ps < 0.2:
                    # Low success rate: Decrease sigma to exploit locally
                    self.sigma = self.sigma * self.k
                
                # Reset counters for the next interval
                self.success_count = 0
                self.total_mutations = 0

        execution_time = time.time() - start_time
        is_successful = 1 if self.best_fitness <= 1e-8 else 0
        
        return {
            "Algorithm_Name": "ES",
            "Function_Name": self.objective_func.__name__,
            "Dimension": self.dim,
            "Run_ID": None,
            "Random_Seed": self.seed,
            "Hyperparameters": {"mu": self.mu, "lambda": self.lambd, "sigma_final": self.sigma},
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