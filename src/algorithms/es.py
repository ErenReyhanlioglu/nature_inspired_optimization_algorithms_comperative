import numpy as np
import time
from src.algorithms.base_optimizer import BaseOptimizer

class ES(BaseOptimizer):
    """
    (mu + lambda) Evolution Strategies (ES) implementation.
    Features dynamic parameter scaling (Schwefel 1/7 rule) and 
    self-adaptive step-size control (1/5 Success Rule).
    """
    def __init__(self, objective_func, bounds, dim, pop_size, max_fes, seed, **kwargs):
        super().__init__(objective_func, bounds, dim, pop_size, max_fes, seed, **kwargs)
        
        if getattr(self, 'lambd', None) is None:
            self.lambd = self.pop_size
            
        # Schwefel's 1/7 Rule
        if getattr(self, 'mu', None) is None:
            self.mu = max(1, self.lambd // 7)
            
        self.hparams['mu'] = self.mu
        self.hparams['lambd'] = self.lambd
        
        range_width = self.bounds[1] - self.bounds[0]
        if getattr(self, 'sigma_init', None) is None:
            self.sigma_init = range_width * 0.1
            
        self.sigma = self.sigma_init
        self.hparams['sigma_init'] = self.sigma_init
        
        # 1/5 Success Rule 
        self.k = 0.85      
        self.update_interval = 10 
        self.success_count = 0
        self.total_mutations = 0

    def optimize(self):
        """
        Executes the (mu + lambda) ES loop with centralized threshold and FE budget control.
        """
        start_time = time.time()
        
        initial_pop = self.initialize_population()
        parents = initial_pop[:self.mu]
        parent_fitness = self.evaluate_fitness(parents)
        
        generation_count = 0
        
        while self.fes_counter < self.max_fes and self.best_fitness > self.success_threshold:
            generation_count += 1
            offspring_population = []
            offspring_parent_indices = []

            for _ in range(self.lambd):
                parent_idx = np.random.randint(0, self.mu)
                parent = parents[parent_idx]

                # Gaussian Mutation: x' = x + N(0, sigma)
                noise = np.random.normal(0, self.sigma, self.dim)
                offspring = parent + noise
                offspring = self.enforce_boundaries(np.atleast_2d(offspring))[0]

                offspring_population.append(offspring)
                offspring_parent_indices.append(parent_idx)

            offspring_population = np.array(offspring_population)
            offspring_fitness = self.evaluate_fitness(offspring_population)

            for i in range(self.lambd):
                self.total_mutations += 1
                if offspring_fitness[i] < parent_fitness[offspring_parent_indices[i]]:
                    self.success_count += 1
            
            combined_pop = np.vstack((parents, offspring_population))
            combined_fit = np.concatenate((parent_fitness, offspring_fitness))
            
            indices = np.argsort(combined_fit)
            parents = combined_pop[indices[:self.mu]]
            parent_fitness = combined_fit[indices[:self.mu]]
            
            # 1/5 Success Rule
            if generation_count % self.update_interval == 0:
                ps = self.success_count / self.total_mutations
                if ps > 0.2:
                    self.sigma = self.sigma / self.k
                elif ps < 0.2:
                    self.sigma = self.sigma * self.k
                
                self.success_count = 0
                self.total_mutations = 0

        execution_time = time.time() - start_time
        
        is_successful = 1 if self.best_fitness <= self.success_threshold else 0
        self.hparams['sigma_final'] = self.sigma
        
        return {
            "Algorithm_Name": "ES",
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