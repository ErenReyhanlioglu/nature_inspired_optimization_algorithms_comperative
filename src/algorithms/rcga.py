import numpy as np
import time
from src.algorithms.base_optimizer import BaseOptimizer

class RCGA(BaseOptimizer):
    """
    Real-Coded Genetic Algorithm (RCGA) implementation.
    Upgraded with SBX (Simulated Binary Crossover) and Per-Gene Gaussian Mutation.
    Inherits administrative functions and success threshold from BaseOptimizer.
    """
    def __init__(self, objective_func, bounds, dim, pop_size, max_fes, seed, **kwargs):
        super().__init__(objective_func, bounds, dim, pop_size, max_fes, seed, **kwargs)
        
        if getattr(self, 'pm', None) is None:
            self.pm = 1.0 / self.dim
        self.hparams['pm'] = self.pm

        self.eta_c = getattr(self, 'eta_c', 20)
        self.hparams['eta_c'] = self.eta_c

    def tournament_selection(self, population, fitness):
        """Selects a single parent using tournament selection logic."""
        indices = np.random.choice(len(population), self.tournament_size, replace=False)
        tournament_fitness = fitness[indices]
        winner_idx = indices[np.argmin(tournament_fitness)]
        return population[winner_idx].copy()

    def optimize(self):
        """
        Executes the RCGA loop with SBX and Per-Gene Mutation tracking.
        """
        start_time = time.time()
        
        population = self.initialize_population()
        fitness = self.evaluate_fitness(population)
        
        while self.fes_counter < self.max_fes and self.best_fitness > self.success_threshold:
            new_population = []
            
            while len(new_population) < self.pop_size:
                p1 = self.tournament_selection(population, fitness)
                p2 = self.tournament_selection(population, fitness)
                
                # --- SBX (Simulated Binary Crossover) ---
                if np.random.rand() < self.pc:
                    u = np.random.rand(self.dim)
                    beta = np.where(u <= 0.5, (2*u)**(1/(self.eta_c+1)), (1/(2*(1-u)))**(1/(self.eta_c+1)))
                    
                    off1 = 0.5 * ((1 + beta) * p1 + (1 - beta) * p2)
                    off2 = 0.5 * ((1 - beta) * p1 + (1 + beta) * p2)
                else:
                    off1, off2 = p1.copy(), p2.copy()
                
                # --- PER-GENE GAUSSIAN MUTATION ---
                for offspring in [off1, off2]:
                    mutation_mask = np.random.rand(self.dim) < self.pm
                    if np.any(mutation_mask):
                        # Mutation strength: 1% of the search space width
                        scale = (self.bounds[1] - self.bounds[0]) * 0.01
                        noise = np.random.normal(0, scale, np.sum(mutation_mask))
                        offspring[mutation_mask] += noise
                    
                    new_population.append(self.enforce_boundaries(np.atleast_2d(offspring))[0])
            
            new_population = np.array(new_population[:self.pop_size])
            new_fitness = self.evaluate_fitness(new_population)

            best_prev_idx = np.argmin(fitness)
            worst_new_idx = np.argmax(new_fitness)
            if fitness[best_prev_idx] < new_fitness[worst_new_idx]:
                new_population[worst_new_idx] = population[best_prev_idx]
                new_fitness[worst_new_idx] = fitness[best_prev_idx]

            population = new_population
            fitness = new_fitness

        execution_time = time.time() - start_time
        is_successful = 1 if self.best_fitness <= self.success_threshold else 0
        
        return {
            "Algorithm_Name": "RCGA",
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